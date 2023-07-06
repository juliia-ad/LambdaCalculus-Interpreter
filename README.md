# AChurch: λ-càlcul interpreter

AChurch is a Lambda Calculus interpreter implemented in Python. Lambda calculus is a formal system introduced by the mathematician Alonzo Church in the 1930s.

## Installation
Python 3.x and antlr4 are required, as well as antlr4.

    pip install antlr4-tools
    pip install antlr4-python3-runtime

If you want to use the Telegram Bot interpreter:

    pip install python-telegram-bot
    pip install pydot
    sudo apt install graphviz

## Getting started
To start using AChurch, run the following commands in order to generate the necessary grammar files:

    antlr4 -Dlanguage=Python3 -no-listener lc.g4
    antlr4 -Dlanguage=Python3 -no-listener -visitor lc.g4

Once the files are ready, you can run the interpreter just by executing:

    python3 achurch.py [num]

`num` lets us choose how we want to launch the interpreter:
- **1** To run the command line interpreter
- **2** To run the telegram bot interpreter

If no number is specified, the command line interpreter will be launched by default. 

In order to use the Telegram Bot, you will need to start a new chat with the AChurch bot of username `@JuliaAChurchBot`.


## Usage
AChurch provides a Lambda Calculus interpreter. Given an expression, it will evaluate it, showing the alpha-conversions and beta-reductions until 
Note: To simplify the input of abstractions, you can use `λ` or `\` to represent the lambda. Same way for the macros, you can define them with `≡` or `=`.

If you are running the interpreter from the Telegram bot, it will also provide an image of the semantic expression tree.

Examples of input:

    ? (λy.x(yz))(ab)
    ? TRUE≡λx.λy.x
    ? NOT≡(λa.((a(λb.(λc.c)))(λd.(λe.d))))
    ? NOT TRUE


## Author
Júlia Amenós Dien

julia.alice.amenos@estudiantat.upc.edu