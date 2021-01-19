;; No backups
(setq backup-inhibited t)
(setq auto-save-default nil)
;; spaces instead of tabs
(setq-default tab-width 4 indent-tabs-mode nil)

;; C style
; use BSD (Allman) style
(setq c-default-style "bsd"
      c-basic-offset 4)
; indentation when hit enter
(eval-after-load 'cc-mode
  '(define-key c-mode-base-map "\C-m" 'newline-and-indent))

;; No tool bar
(tool-bar-mode 0)

;; No startup message
(setq inhibit-startup-message t)

;; Color theme
(load-theme 'deeper-blue t)

;; ido
(ido-mode 1)
(ido-everywhere 1)
(setq ido-enable-flex-matching t)

(setq initial-frame-alist
      (append (list
               '(width . 100)
               '(height . 40))
              initial-frame-alist))
(setq default-frame-alist initial-frame-alist)

;; Silent (either of the following)
(setq visible-bell t)
(setq ring-bell-function 'ignore)
