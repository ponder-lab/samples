# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
import hashlib
import math
import os.path
import random
import re
import sys
import tarfile

import numpy as np
import urllib
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185

def prepare_words_list(wanted_words):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


def get_features_range(model_settings):
  """Returns the expected min/max for generated features.

  Args:
    model_settings: Information about the current model being trained.

  Returns:
    Min/max float pair holding the range of features.

  Raises:
    Exception: If preprocessing mode isn't recognized.
  """
  # TODO(petewarden): These values have been derived from the observed ranges
  # of spectrogram and MFCC inputs. If the preprocessing pipeline changes,
  # they may need to be updated.
  if model_settings['preprocess'] == 'average':
    features_min = 0.0
    features_max = 127.5
  elif model_settings['preprocess'] == 'mfcc':
    features_min = -247.0
    features_max = 30.0
  elif model_settings['preprocess'] == 'micro':
    features_min = 0.0
    features_max = 26.0
  else:
    raise Exception('Unknown preprocess mode "%s" (should be "mfcc",'
                    ' "average", or "micro")' % (model_settings['preprocess']))
  return features_min, features_max


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
               wanted_words, validation_percentage, testing_percentage,
               model_settings, summaries_dir):
    if data_dir:
      self.data_dir = data_dir
      self.maybe_download_and_extract_dataset(data_url, data_dir)
      self.prepare_data_index(silence_percentage, unknown_percentage,
                              wanted_words, validation_percentage,
                              testing_percentage)
      self.prepare_background_data()
    #self.prepare_processing_graph(model_settings, summaries_dir)

  def maybe_download_and_extract_dataset(self, data_url, dest_directory):
    """Download and extract data set tar file.

    If the data set we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a
    directory.
    If the data_url is none, don't download anything and expect the data
    directory to contain the correct files already.

    Args:
      data_url: Web location of the tar file containing the data set.
      dest_directory: File path to extract data to.
    """
    if not data_url:
      return
    if not gfile.Exists(dest_directory):
      os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not gfile.Exists(filepath):

      def _progress(count, block_size, total_size):
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' %
            (filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      try:
        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
      except:
        tf.compat.v1.logging.error(
            'Failed to download URL: {0} to folder: {1}. Please make sure you '
            'have enough free space and an internet connection'.format(
                data_url, filepath))
        raise
      print()
      statinfo = os.stat(filepath)
      tf.compat.v1.logging.info(
          'Successfully downloaded {0} ({1} bytes)'.format(
              filename, statinfo.st_size))
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index + 2
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    search_path = os.path.join(self.data_dir, '*', '*.wav')
    for wav_path in gfile.Glob(search_path):
      _, word = os.path.split(os.path.dirname(wav_path))
      word = word.lower()
      # Treat the '_background_noise_' folder as a special case, since we expect
      # it to contain long audio samples we mix in to improve training.
      if word == BACKGROUND_NOISE_DIR_NAME:
        continue
      all_words[word] = True
      set_index = which_set(wav_path, validation_percentage, testing_percentage)
      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      if word in wanted_words_index:
        self.data_index[set_index].append({'label': word, 'file': wav_path})
      else:
        unknown_index[set_index].append({'label': word, 'file': wav_path})
    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(wanted_words):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_index['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      random.shuffle(unknown_index[set_index])
      unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
      self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

  def prepare_background_data(self):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """
    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not gfile.Exists(background_dir):
      return self.background_data

    search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                               '*.wav')
    for wav_path in gfile.Glob(search_path):
      wav_loader = io_ops.read_file(tf.constant(wav_path, dtype=tf.string))
      wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
      wav_data = wav_decoder.audio.numpy().flatten()
      self.background_data.append(wav_data)
    if not self.background_data:
      raise Exception('No background wav files were found in ' + search_path)

  #def prepare_processing_graph(self, model_settings, summaries_dir):
  def get_output_audio(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_padding_placeholder_: Where to pad the clip.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - output_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
      summaries_dir: Path to save training summary information to.

    Raises:
      ValueError: If the preprocessing mode isn't recognized.
      Exception: If the preprocessor wasn't compiled in.
    """
    #with tf.compat.v1.get_default_graph().name_scope('data'):
    desired_samples = model_settings['desired_samples']


    #self.wav_filename_placeholder_ = tf.compat.v1.placeholder(
    #    tf.string, [], name='wav_filename')


    wav_loader = io_ops.read_file(self.wav_filename_placeholder_)
    wav_decoder = tf.audio.decode_wav(
        wav_loader, desired_channels=1, desired_samples=desired_samples)



    # Allow the audio sample's volume to be adjusted.
    #self.foreground_volume_placeholder_ = tf.compat.v1.placeholder(
    #    tf.float32, [], name='foreground_volume')



    scaled_foreground = tf.multiply(wav_decoder.audio,
                                    self.foreground_volume_placeholder_)


    # Shift the sample's start position, and pad any gaps with zeros.
    #self.time_shift_padding_placeholder_ = tf.compat.v1.placeholder(
    #    tf.int32, [2, 2], name='time_shift_padding')
    #self.time_shift_offset_placeholder_ = tf.compat.v1.placeholder(
    #    tf.int32, [2], name='time_shift_offset')


    padded_foreground = tf.pad(
        tensor=scaled_foreground,
        paddings=self.time_shift_padding_placeholder_,
        mode='CONSTANT')
    sliced_foreground = tf.slice(padded_foreground,
                                 self.time_shift_offset_placeholder_,
                                 [desired_samples, -1])



    # Mix in background noise.
    #self.background_data_placeholder_ = tf.compat.v1.placeholder(
    #    tf.float32, [desired_samples, 1], name='background_data')
    #self.background_volume_placeholder_ = tf.compat.v1.placeholder(
    #    tf.float32, [], name='background_volume')



    background_mul = tf.multiply(self.background_data_placeholder_,
                                 self.background_volume_placeholder_)
    background_add = tf.add(background_mul, sliced_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    spectrogram = audio_ops.audio_spectrogram(
        background_clamp,
        window_size=model_settings['window_size_samples'],
        stride=model_settings['window_stride_samples'],
        magnitude_squared=True)



    #tf.compat.v1.summary.image(
    #    'spectrogram', tf.expand_dims(spectrogram, -1), max_outputs=1)



    # The number of buckets in each FFT row in the spectrogram will depend on
    # how many input samples there are in each window. This can be quite
    # large, with a 160 sample window producing 127 buckets for example. We
    # don't need this level of detail for classification, so we often want to
    # shrink them down to produce a smaller result. That's what this section
    # implements. One method is to use average pooling to merge adjacent
    # buckets, but a more sophisticated approach is to apply the MFCC
    # algorithm to shrink the representation.
    if model_settings['preprocess'] == 'average':
      self.output_ = tf.nn.pool(
          input=tf.expand_dims(spectrogram, -1),
          window_shape=[1, model_settings['average_window_width']],
          strides=[1, model_settings['average_window_width']],
          pooling_type='AVG',
          padding='SAME')
      tf.compat.v1.summary.image('shrunk_spectrogram',
                                 self.output_,
                                 max_outputs=1)
    elif model_settings['preprocess'] == 'mfcc':
      self.output_ = audio_ops.mfcc(
          spectrogram,
          wav_decoder.sample_rate,
          dct_coefficient_count=model_settings['fingerprint_width'])
      #tf.compat.v1.summary.image(
      #    'mfcc', tf.expand_dims(self.output_, -1), max_outputs=1)
    elif model_settings['preprocess'] == 'micro':
      if not frontend_op:
        raise Exception(
            'Micro frontend op is currently not available when running'
            ' TensorFlow directly from Python, you need to build and run'
            ' through Bazel')
      sample_rate = model_settings['sample_rate']
      window_size_ms = (model_settings['window_size_samples'] *
                        1000) / sample_rate
      window_step_ms = (model_settings['window_stride_samples'] *
                        1000) / sample_rate
      int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)
      micro_frontend = frontend_op.audio_microfrontend(
          int16_input,
          sample_rate=sample_rate,
          window_size=window_size_ms,
          window_step=window_step_ms,
          num_channels=model_settings['fingerprint_width'],
          out_scale=1,
          out_type=tf.float32)
      self.output_ = tf.multiply(micro_frontend, (10.0 / 256.0))
      #tf.compat.v1.summary.image(
      #    'micro',
      #    tf.expand_dims(tf.expand_dims(self.output_, -1), 0),
      #    max_outputs=1)
    else:
      raise ValueError('Unknown preprocess mode "%s" (should be "mfcc", '
                       ' "average", or "micro")' %
                       (model_settings['preprocess']))

    ## Merge all the summaries and write them out to /tmp/retrain_logs (by
    ## default)
    #self.merged_summaries_ = tf.compat.v1.summary.merge_all(scope='data')
    #if summaries_dir:
    #  self.summary_writer_ = tf.compat.v1.summary.FileWriter(
    #      summaries_dir + '/data', tf.compat.v1.get_default_graph())

  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, model_settings, background_frequency,
               background_volume_range, time_shift, mode):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
      time_shift: How much to randomly shift the clips by in time.
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of label indexes

    Raises:
      ValueError: If background samples are too short.
    """
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    if how_many == -1:
      sample_count = len(candidates)
    else:
      sample_count = max(0, min(how_many, len(candidates) - offset))
    # Data and labels will be populated and returned.
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)
    desired_samples = model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')
    pick_deterministically = (mode != 'training')
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in range(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i
      else:
        sample_index = np.random.randint(len(candidates))
      sample = candidates[sample_index]
      # If we're time shifting, set up the offset for this sample.
      if time_shift > 0:
        time_shift_amount = np.random.randint(-time_shift, time_shift)
      else:
        time_shift_amount = 0
      if time_shift_amount > 0:
        time_shift_padding = [[time_shift_amount, 0], [0, 0]]
        time_shift_offset = [0, 0]
      else:
        time_shift_padding = [[0, -time_shift_amount], [0, 0]]
        time_shift_offset = [-time_shift_amount, 0]




      #input_dict = {
      #    self.wav_filename_placeholder_: sample['file'],
      #    self.time_shift_padding_placeholder_: time_shift_padding,
      #    self.time_shift_offset_placeholder_: time_shift_offset,
      #}
      self.wav_filename_placeholder_ = tf.constant(sample['file'], tf.string)
      self.time_shift_padding_placeholder_ = tf.constant(time_shift_padding,
                                                         tf.int32)
      self.time_shift_offset_placeholder_ = tf.constant(time_shift_offset,
                                                        tf.int32)


      
      # Choose a section of background noise to mix in.
      if use_background or sample['label'] == SILENCE_LABEL:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        if len(background_samples) <= model_settings['desired_samples']:
          raise ValueError(
              'Background sample is too short! Need more than %d'
              ' samples but only %d were found' %
              (model_settings['desired_samples'], len(background_samples)))
        background_offset = np.random.randint(
            0, len(background_samples) - model_settings['desired_samples'])
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples)]
        background_reshaped = background_clipped.reshape([desired_samples, 1])
        if sample['label'] == SILENCE_LABEL:
          background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random.uniform(0, background_volume_range)
        else:
          background_volume = 0
      else:
        background_reshaped = np.zeros([desired_samples, 1])
        background_volume = 0



      #input_dict[self.background_data_placeholder_] = background_reshaped
      #input_dict[self.background_volume_placeholder_] = background_volume
      self.background_data_placeholder_ = tf.constant(background_reshaped,
                                                      tf.float32)
      self.background_volume_placeholder_ = tf.constant(background_volume,
                                                        tf.float32)



      # If we want silence, mute out the main sample but leave the background.
      if sample['label'] == SILENCE_LABEL:
        #input_dict[self.foreground_volume_placeholder_] = 0
        self.foreground_volume_placeholder_ = tf.constant(0, tf.float32)
      else:
        #input_dict[self.foreground_volume_placeholder_] = 1
        self.foreground_volume_placeholder_ =  tf.constant(1, tf.float32)




      # Run the graph to produce the output audio.
      #summary, data_tensor = sess.run(
      #    [self.merged_summaries_, self.output_], feed_dict=input_dict)
      #self.summary_writer_.add_summary(summary)

      self.get_output_audio(model_settings)
      data_tensor = self.output_
      
      data[i - offset, :] = data_tensor.numpy().flatten()
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
    return data, labels



def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, feature_bin_count,
                           preprocess):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    feature_bin_count: Number of frequency bins to use for analysis.
    preprocess: How the spectrogram is processed to produce features.

  Returns:
    Dictionary containing common settings.

  Raises:
    ValueError: If the preprocessing mode isn't recognized.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if preprocess == 'average':
    fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
    average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
    fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
  elif preprocess == 'mfcc':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  elif preprocess == 'micro':
    average_window_width = -1
    fingerprint_width = feature_bin_count
  else:
    raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                     ' "average", or "micro")' % (preprocess))
  fingerprint_size = fingerprint_width * spectrogram_length
  return {
      'desired_samples': desired_samples,
      'window_size_samples': window_size_samples,
      'window_stride_samples': window_stride_samples,
      'spectrogram_length': spectrogram_length,
      'fingerprint_width': fingerprint_width,
      'fingerprint_size': fingerprint_size,
      'label_count': label_count,
      'sample_rate': sample_rate,
      'preprocess': preprocess,
      'average_window_width': average_window_width,
  }

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to randomly shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=int,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is.',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between spectrogram timeslices.',
  )
  parser.add_argument(
      '--feature_bin_count',
      type=int,
      default=40,
      help='How many bins to use for the MFCC fingerprint',
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--check_nans',
      type=bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--quantize',
      type=bool,
      default=False,
      help='Whether to train the model for eight-bit deployment')
  parser.add_argument(
      '--preprocess',
      type=str,
      default='mfcc',
      help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')

  # Function used to parse --verbosity argument
  def verbosity_arg(value):
    """Parses verbosity argument.

    Args:
      value: A member of tf.logging.
    Raises:
      ArgumentTypeError: Not an expected value.
    """
    value = value.upper()
    if value == 'DEBUG':
      return tf.compat.v1.logging.DEBUG
    elif value == 'INFO':
      return tf.compat.v1.logging.INFO
    elif value == 'WARN':
      return tf.compat.v1.logging.WARN
    elif value == 'ERROR':
      return tf.compat.v1.logging.ERROR
    elif value == 'FATAL':
      return tf.compat.v1.logging.FATAL
    else:
      raise argparse.ArgumentTypeError('Not an expected value')
  parser.add_argument(
      '--verbosity',
      type=verbosity_arg,
      default=tf.compat.v1.logging.INFO,
      help='Log verbosity. Can be "DEBUG", "INFO", "WARN", "ERROR", or "FATAL"')
  parser.add_argument(
      '--optimizer',
      type=str,
      default='gradient_descent',
      help='Optimizer (gradient_descent or momentum)')

  FLAGS, unparsed = parser.parse_known_args()
  #tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

  model_settings = prepare_model_settings(
      len(prepare_words_list(FLAGS.wanted_words.split(','))),
      FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
      FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess)
  audio_processor = AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir,
      FLAGS.silence_percentage, FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
      FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir)

  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

  train_fingerprints, train_ground_truth = audio_processor.get_data(
      FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
      FLAGS.background_volume, time_shift_samples, 'training')

  validation_fingerprints, validation_ground_truth = \
      audio_processor.get_data(FLAGS.batch_size, 0, model_settings, 0.0,
                               0.0, 0, 'validation')
