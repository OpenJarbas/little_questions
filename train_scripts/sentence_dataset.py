from os.path import join, dirname
from pprint import pprint

DATA_PATH = join(dirname(__file__), "data")

dataset = []

yesno = [
    "can you die",
    "will you live forever",
    "Are you a student",
    "Does Jessica like history lectures?",
    "Doesn’t Jessica like history lectures?",
    "Did he play on a football team?",
    "Didn’t he play on a football team?",
    "Will the teacher be late",
    "should i program artificial stupidity",
    "did you know that dogs are animals",
    "do you agree that life is beautiful",
    "have you finished booting up",
    "Do you like Paris",
    "Can you speak Russian",
    "Will you marry me",
    "Have you ever thought how much benefit school uniforms can bring to students",

]
with open(join("clean_data", "yes_no.txt")) as f:
    dataset += ["QUESTION:YESNO " + y.rstrip(" ")
                for y in f.read().split("\n") + yesno]

questions = [
    "Where is the cat",
    "What time will you finish writing your English homework?",
    "What is an adverb?",
    "How often do you read this article?",
    "What did you have for breakfast",
    "When did the dinosaurs live",
    "What is your favorite movie",
    "Which newspaper do you read",
    "Who is your favourite actor",
    "who made you",
    "when will the world end",
    "how fast can an elephant run",
    "why are fire trucks red",
    "what do dogs and cats have in common",
    "what is a living being",
    "what is the speed of light",
    "when is your birthday",
    "when were you born",
    "where do you store your data",
    "who made you",
    "how long until world war 3",
    "how long ago was sunrise",
    "which city has more people",
    "who made you",
    "whose dog is this",
    "how much is bitcoin worth",
    "What is it",
    "What is expensive",
    "What has been ordered",
    "What do you like",
    "What is Peter doing",
    "What can we speak",
    "What was she washing",
    "What did you buy",
    "What is she",
    "What will he buy",
    "What is he playing",
    "What's he playing",
    "What does he teach",
    "What should we buy",
    "What did you eat",
    "What do you study",
    "What day is Christmas",
    "what time does the class start/When does the class start",
    "What does huge mean",
    "Why is she crying",
    "Why was she crying",
    "Why did she call you",
    "Whose car is that",
    "Whose website is this",
    "Whose bag is on the table",
    "Whose bag's on the table",
    "Who is she",
    "Who does John like",
    "Who can cook well",
    "Who likes Lisa",
    "Who are they",
    "Who said that",
    "Who did she call",
    "Who is coming",
    "Who's coming",
    "Who knows the answer",
    "Who is sleeping",
    "Where is Vancouver",
    "Where has he visited",
    "Where did they go",
    "Where are you going tomorrow",
    "Where is your home district",
    "When is your birthday",
    "When can she come",
    "When is Christmas",
    "When does the class start",
    "When can we meet",
    "What kind of music do you like",
    "What kind of dog do you have"

]
dataset += ["QUESTION:QUERY " + y.rstrip(" ") for y in questions]

"""
A statement is defined as having a structure in which there is typically a Subject,
followed by a verb and then a further unit such as a Direct Object.
For example,
    Jimmy loves his dog,
    The government will make an announcement at noon,
    She reads two newspapers every day

"""
statements = [
    "Jimmy loves his dog",
    "I am a student",
    "Dinosaurs lived millions of years ago",
    "This is my favorite movie",
    "The government will make an announcement at noon",
    "She reads two newspapers every day",
    "I like pizza",
    "I want you to buy bitcoin",
    "not a question",
    "I feel terrible",
    "We own a cat",
    "Jessica likes history lectures",
    "Jessica does not like history lectures",
    "He plays on a football team",
    "He doesn’t play on a football team",
    "She wrote",
    "He scored a goal.",
    "I completed my college application essay.",
    "Peanut is better than jam.",
    "Students failed to complete their essays on time.",
    "My wife loves eating cake in the morning.",
    "The developer needs new resources for completing a project."
    "She completed her literature review",
    "He organized his sources by theme",
    "They studied APA rules for many hours",
    "She completed her literature review, and she created her reference list",
    "He organized his sources by theme; then, he updated his reference list",
    "They studied APA rules for many hours, but they realized there was still much to learn"
]
dataset += ["SENTENCE:STATEMENT " + y.rstrip(" ") for y in statements]

"""
Commands also have a special structure in that they typically lack a Subject.
Examples are:
    Eat your dinner
    Be quiet
    Open the door, etc.

Not all imperative sentences are orders or commands.
They can be social expressions.
    Have a nice day.
    Get well soon.
    Help yourselves to coffee.
"""
commands = [
    "Welcome the new student",
    "stop",
    "Open the pod bay doors",
    "tell me about evil",
    "Read this book now.",
    "give examples of animals",
    "Eat your dinner",
    "Be quiet",
    "Open the door",
    "Feed the cat",
    "Please get me dinosaur socks",
    "Play the movie",
    "Shut the door",
    "Bob, feed the cat",
    "Brush your teeth",
    "Mom, Please get me dinosaur socks",
    "Jenny, Play the movie",
    "Attend history lectures",
    "Do not attend history lectures",
    "Join a football team",
    "Don’t join a football team",
    "Please leave your shoes outside",
    "Do not stop!",
    "Never speak to me like that again"
]
social = [
    "Have a nice day",
    "Get well soon",
    "Help yourselves to coffee"
]
dataset += ["SENTENCE:SOCIAL " + y.rstrip(" ") for y in social]
dataset += ["QUESTION:COMMAND " + y.rstrip(" ") for y in commands]

"""
We can make a request, which is a type of command,
sound more polite by using the interrogative.
   Would you feed the dog, please.
   Would you mind shutting the door.
   Could I have that now, thank you.
"""
requests = [
    "Could you pass me the salt please",
    "Would you feed the dog, please",
    "Would you mind shutting the door",
    "Could I have that now, thank you"
]
dataset += ["QUESTION:REQUEST " + y.rstrip(" ") for y in requests]

"""
Exclamations grammatically have a structure that involves the words what a or how,

    What a nice person you are!
    What a beautiful painting!,
    How clever you are!,
    How wonderful that is!

(Notice that the Subject goes before the verb in How clever you are!
If this were a question we would have How clever are you?)

"""
exclamation = [
    "What big eyes you have",
    "What big ears you have",
    "There are so many students here!",
    "What a nice dog you have there!",
    "What a nice person you are!",
    "What a beautiful painting!",
    "How clever you are!",
    "What a cute dog!",
    "How wonderful that is!",
    "What a terrible, big mouth you have!",
    "What a beautiful painting!",
    "What an excellent idea it was to throw him a surprise party!",
    "How nice it was!"
]
dataset += ["SENTENCE:EXCLAMATION " + y.rstrip(" ") for y in statements]

dataset = sorted(list(set(dataset)))
pprint(dataset)

with open(join("clean_data", "simple_tags_EN_0.7.0a1.txt"), "w") as f:
    f.write("\n".join(dataset))
