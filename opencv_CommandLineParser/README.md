opencv_CommandLineParser

## keys string

define options. format is the following,

    { argument name | default value | description }

If default value is empty the function has() can check if empty.
If default value is '<none>' the returned string must not be empty.

##positional argument

argument to be placed in order
put '@' before the argument name in keys string

##optional argument

argument to be placed as an option

    > command -opt=aaa
    > command -opt

