## About
A markov chain based text generator. The python file reads a sample text, and generates new text based on the transition probabilities assigned to each word.

## Explanation
Markov chains are a way of determining the future state of a system based on the current state. Different states can be connected by a transition probability; that is the probability that the state will transition from the current state to a given next state. If a word, or collection of words is considered to be a state, an example sentence could be visualized as a markov chain:

### Sentence
"My name is Sherlock Holmes. It is my buisiness to know what other people do not know."

### Markov Chain Visualization
(Credit to Normalized Nerds video on the subject for this diagram - https://www.youtube.com/watch?v=E4WcBWuQQws&t=139s)

Given a starting point on the diagram, you can randomly walk around the diagram based on the probabilities, chaining together words until you have a sentence. 

## Build
Clone the repository and run the main.py script with the requirements listed in the requirements.txt file. These can be installed by running `python pip install -r requirements.txt`. It is recommended to set up a virtual environment in the repository to install the packages to. Works with python version 3.11.2 and later.

## Usage
Invoke the script by running `python main.py [-args]`. Valid arguments are provided below:

    options:
      -h, --help         show this help message and exit
      -t T, -training T  Path to a training text file. Defaults to the default.txt in the repo directory.
      -s S, -span S      The max number of words to group together into a token. Higher span may reduce chain accuracy for shorter texts, but allows for more complex output.
      -l L, -length L    Length of the output text
      -p P, -prompt P    Prompt to generate text from
An example usage would be:

    python .\main.py -t "default_long.txt" -s 2 -l 300 -p "There was a dark and frightening sun"