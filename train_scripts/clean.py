from os.path import join, dirname
from pprint import pprint

DATA_PATH = join(dirname(__file__), "data")

questions = []
with open(join(DATA_PATH, "questions.txt")) as f:
    questions += f.read().split("\n")
with open(join(DATA_PATH, "questions_test.txt")) as f:
    questions += f.read().split("\n")

questions = [
    " ".join([w for w in q.rstrip(" ?").replace('"', "").replace(": ", "")
             .replace(" ,", ",").replace(" 's", "'s").replace("`", "'")
             .split(" ") if w])
    for q in questions]
questions = sorted(list(set(questions)))
pprint(questions)

with open(join("clean_data", "raw_questions.txt"), "w") as f:
    f.write("\n".join(questions))

# extended
animal = [
    "what is the smallest bird and the smallest warm-blooded vertebrate",
    "What is the deadliest animal in the world",
    "what is the biggest predator in the animal kingdom",
    "Which animal gives birth standing up",
    "what is The slowest Animals In The World",
    "What is the heaviest and tallest Penguin",
    "What is the fastest land animal",
    "What is the loudest animals in the world",
    "What is the friendliest animal in the world",
    "What is the largest cat species in the world",
    "what is the largest land carnivorous animals in the world",
    "What is the friendliest dog breeds in the world",
    "What is the largest dogs in the world",
    "Which animal has the Strongest Bite in the Animal Kingdom",
    "What animal is pregnant for the shortest time",
    "Which animal has the longest gestation (pregnancy) period of all mammals",
]
body_part = [
    "Which set of body parts is found in the upper limbs",
    "Which body parts work together when one is watching TV",
    "Which set of body parts are found in the head",
    "What do we call to the ''main division of the body''  where waist, belly button, abdomen and chest are found",
    "How many bones are there in the human foot",
    "What are the largest salivary glands",
    "what are the 3 types of muscles in the body",
    "Which organ has connections with seven cavities which are 2 nasal cavities, 2 tympanic cavities (cavity which surrounds the bones present in the middle ear), mouth, esophagus, larynx",
    "Which body part contains the smallest muscles in the body alongside the smallest bones",
    "Which muscles are the only muscles that can be consciously controlled",
    "What are made of elastic tissue and also play a key role in the functioning of joints. They connect muscle to bone",
    "Which of the listed organs are not related to the urination system: bladder, kidneys, pharynx, prostate, ureters, urethra",
    "What are short bands of tough fibrous connective tissue that function to connect one bone to another, forming the joint",
    "LIke fingerprint, which is the other part of the body which is unique",
    "Bile is released from what organ",
    "What is the largest part of the human brain is called",
    "What is a vital internal organ and gland, which carries out over 500 functions",
    "What digestive organ concentrates the bile into the form that's best used for digestion",
    "What is the rarest blood type",
    "Which body system is home to more cancers, and causes more cancer mortalities, than any other organ system in the body",
    "Which is the only type of muscle that can be consciously controlled",
    "What is the strongest and longest bone in the human body",
    "What is the largest organ of the human body",
    "What is the strongest muscle in the human body",
    "What is the hardest-working muscle in the body",
    "What is the largest internal organ of the human body",
    "What part of the human body doesn't change its size during a person's lifetime",
    "What is the body’s largest muscle in the buttocks and helps humans maintain an upright posture",
    "Which muscle lines the inside of blood vessels and organs, such as the stomach, and is also known as visceral muscle",
    "What of the body liquid contains immune cells, antimicrobial and antifungal proteins, and growth factors that promote wound healing",
    "What helps hold neurons in place, supply nutrients to neurons, destroy germs, remove dead neurons, and direct axons of neurons",
    "Which type of hair develops from childhood covering most of the human body, it is a short, fine, light-colored hair that is often barely noticeable",
    "Which type of neurons transmits neural signals to activate muscles or glands?",
    "Which organ is made up of four chambers, the left atrium, right atrium, left ventricle, and right ventricle",
    "The part of the eye that allows us to focus on different things in known as what",
    "Which gland secretes mucus and is found in the duodenum only",
    "Which organ releases bicarbonate and digestive enzymes such as trypsin, lipase, and amylase during the digestion process",
    "Which organ of the digestive system is largely responsible for the breakdown of food in the small intestine, using enzymes",
    "What is the only muscle in the human body that works without any support from the skeleton",
    "Which of the digestive organ absorbs the nutrients of food and passes them into the bloodstream",
    "Buccal cavity (oral cavity) and the nasal passage opens into which organ, nearly 12.5 cm long or 5 inches long",
    "Which organ in the mouth maintains balance of the middle ear and send the inhaled air to the wind pipe or trachea"
]
food = [
    "Which is the most expensive spice in the world by weight",
    "Which Asian fruit has the nickname ‘king of fruits’ and is known for its distinctive smell",
    "Which Mexican food has a name meaning “little donkey”",
    "What is the most stolen food in the world",
    "What was the main dish at Medieval Christmas feasts",
    "Which is the only edible food that never goes bad",
    "What type of beans are used to make baked beans",
    "Most people make a homemade baked version of which dish for Thanksgiving",
    "Which cheese is known as the King of English cheeses",
    "What falling fruit supposedly inspired Isaac Newton to write the laws of gravity",
    "In France, what food item literally means 'twice cooked'",
]
religion = [
    "What religious family do you belong to or identify yourself most close to",
    "What is your religion",
    "what was the religion of the ancient peoples who inhabited north-central Anatolia",
    "what was the religion of the ancient greeks",
    "what was the religion of the roman republic",
    "what religion is practiced in south africa",
    "what is called the religion of peace",
    "what religion do the jews practice",
    "which religions believe in purgatory",
    "what religions have the concept of hell",
    "what religions believe in heaven",
    "what was the religion of the celts",
    "what religion did the egyptians practice",
    "which religion worships the most deities",
    "which religion has the most gods",
    "name all the religions still practiced in scandinavia",
    "what polytheistic religions are still practiced today",
    "what was the religion in ancient iberia",
    "name all the religions practiced around the Mediterranean sea",
    "what religion am I",
    "Ever wondered what religion you are"
]
temp = [
    "what is the temperature at the earth's core",
    "A general rule of thumb used by pilots is for every 1,000 feet of altitude, the temperature falls 3.5 F. If the temperature at sea level is 78 F, what would you expect the temperature to be at 10,000 feet",
    "Which temperature is hotter: 17 C or 58 F?",
    "Pure iron melts at 1,535 C. What is the temperature in Fahrenheit",
    "Oxygen has a boiling point of 90.19 K. What is the temperature in Fahrenheit",
    "at what temperature does water freeze",
    "The average surface temperature on Mars is -63 C. What is the temperature in Fahrenheit",
    "Room temperature is often used in calculations as 300 K. What is the temperature in Fahrenheit",
    "at what temperature does water boil",
    "The title of the book 'Fahrenheit 451' refers to the temperature that book paper burns, or 451 F. What is the temperature in Celsius (to the hundredth)",
    "Body temperature is 98.6 F. What is the temperature in Celsius",
    "Gallium is a metal that can melt in your hand at 302.93 K. What is the temperature in Celsius (to the hundredth)",
    "at what temperature does water freeze",
    "What is the surface temperature of the sun",
    "Aluminum metal melts at 660.37 C. What is the temperature in Kelvin (to the hundredth)",
    "What will be the final temperature of water that was initially 40°C but was mixed with water that was 20 °C"
]
vol = [
    "how big was the roman empire",
    "what is the volume of the sun",
    "How many cubic units",
    "What is the volume of the figure in cubic units",
    "Each block on the tower shown measures 4 cubic cm in volume. What is the total volume of the tower",
    "what is the size of jupiter",
    "The surface area of a cube is 1734 sq. cm. Find its volume",
    "what is the volume of the observable universe",
    "what is the volume of the milky way",
    "how large are the pillars of creation",
    "A cube measures 4 cm on a side. What is the volume of the cube",
    "how big was the universe 1 second after the big bang",
    "what size will the sun be when it becomes a red dwarf",
    "what is the size of the moon",
    "how small is a microbe",
    "what is the size, in cubic centimeters, of a tardigrade",
    "what is the size of the smallest organism",
    "what size was the largest reptile to ever live",
    "what was the size of an average female t-rex",
    "what size do you wear",
    "how small is the smallest organism",
    "what is the cubic measurement of the cybertruck",
    "how many cubic meters does a whale's stomach have",
    "how many cubic millimeters is a red blood cell",
    "how many cubic centimeters in an average wine bottle",
    "how large is Oumuamua",
    "how big is the Enterprise"
]
speed = ["how fast can a human run",
         "what is the speed of darkness",
         "do you know what is the top speed of a U-boat",
         "how fast can a snake move",
         "how fast is voyager going",
         "how slow does a turtle move",
         "how slow does a sloth move",
         "what is the velocity of sound",
         "what is the average velocity of a horse",
         "what is the average speed of a komodo dragon",
         "do you know the speed of light in a vaccum"]
weight = [
    "how much does an elephant weigh",
    "what is the mass of the sun",
    "he excavator bucket capacity is 0.5 m3. Determine the mass of sand that the excavator picks up",
    "Calculate how many grams of oxygen are in 50 g of carbon dioxide CO2",
    "what is the mass of the moon",
    "how many kilograms in 1 teaspoon of neutron star",
    "how many grams of sugar do you put in your coffee",
    "how many kg in a typical brick",
    "how much does a hippo weigh",
    "how many kilograms",
    "how much does the cybertruck weigh",
    "Calculate the mass of the cuboid with dimensions of 12 cm; 0.8 dm and 100 mm made from spruce wood ",
    "how much do you weigh in pounds",
    "what is your body mass",
    "what is the atomic mass of Gallium",
    "What is the mass of 500 m of copper wire with a diameter of 1 mm",
    "Dinesh ate 3 cookies, each with mass 1.45 g. What was the total mass of the cookies that he ate",
    "on average, how many grams in an apple",
    "Barrel of oil weighs 283 kg. When it mold 26% oil, weighed 216 kg. What is the mass of the empty barrel",
    "how many kilos does a car weight on average",
    "what is the mass of the photon",
    "Half the weight of a brick plus 20 pounds is equal to 1/3 the weight of the brick plus 30 pounds. How much does the brick weigh",
    "An empty can has a mass of 1/6 lb. When it is filled with sand, it has a mass of 7/12 lb. Find the mass of the sand in the can",
    "What is the weight of a 40 kg child on Mars",
    "A car has a weight of 10,000 N. What is the mass of the car on Earth",
    "what is your weight",
    "how many tons of food do humans waste annually",
    "how many tons of water do domesticated animals drink annually",
    "How many tons of molten bell metal and how many tons of copper is needed to make 100 tons of Gunmetal",
    "The whole loaded train weighs 360 tons. It has twenty wagons, each carrying 12 tons of grain. What is the weight of the locomotive",
    "How many grams of salt should we dissolve in 400 g of water to get a 20% solution",
    "The vanillin molecule is the primary molecule present in vanilla extract. What is the molecular mass of vanillin",
    "what is the weight of 5 liters of water",
    "Each cement block weighs 7.9 kilograms. How much do 6 blocks weigh in total",
    "what is the combined mass of all ants in the world",
    "5 of the same bread has the same weight as three bread and 4 kg of fruit. What weight has one bread",
    "Meat loses 18% of its weight by smoking. How much raw meat butcher used to manufacture 35 kilos of smoked",
    "what is the mass of the black hole in the center of the milky way",
    "tell me the mass of jupiter",
    "Cacao contains 34% filling. How many grams of filling are in 130 g cacao.",
    "One kilogram equals 2.2 pounds. If a patient weighs 79.5kg, what is his weight in pounds",
    "A recipe requires 2 pounds of flour. If a chef wants to triple the recipe, how many ounces of flour will be needed"
]
color = [
    "is the Sun white or yellow",
    "is the ocean blue or green",
    "What is the colour of the circle which is the sixth (6th) from the left",
    "A hyperbolic color may be seen by staring at a bright color and then viewing its complementary color. For example, staring at magenta produces a green afterimage. If you stare at magenta and then look at something green, the afterimage is what color?",
    "Self-luminous colors appear to glow even though no light is emitted. An example is 'self-luminous red' which may be seen by staring at green and then looking at white. Do you know any other Self-luminous color?",
    "what if your favorite colour",
    "what colour is the sky",
    "Stygian colors are dark and supersaturated. For example, 'stygian blue' may be seen by staring at bright yellow and then looking at black. The normal afterimage is dark blue. When viewed against black, the resulting blue is as dark as black, yet colored. Does stygian white also exist?",
    "what is the color of blood",
    "is blood blue or red",
    "what colors can a blind person see",
    "name 5 impossible colors",
    "human eye perceives white as a mixture of which different spectral colors",
    "is your favorite color black, white, blue or green",
    "what is your lucky colour",
    "what color comes after violet in the visible spectrum",
    "what colour comes before yellow in the visible spectrum",
    "what primary colours should you mix to get brown",
    "if Stygian blue is a real color, what other Stygian colors exist",
    "staring at bright yellow causes a dark blue afterimage, then on looking at black, the blue is seen as blue against the black, also as dark as the black, what is this color named"
]
num_code = [
    "What is Kevin's phone number",
    "what is mycroft's headquarters zip code",
    "what is my phone number",
    "what is dad's number",
    "what number should i call in case of emergency",
    "tell me the hospital's phone number",
    "what is my pairing code",
    "what is your public PGP key",
    "what's the decryption code",
    "what number decrypts this secret message",
    "what is my personal code",
    "what is the pin number",
    "what is the secret mission's code number",
    "tell me the secret code to open the safe",
    "what is the number that unlocks the vault",
    "what number corresponds to your secret key",
    "give me your private bitcoin wallet key",
    "tell me the pin to unlock the phone",
    "i need to know the zip code to place an order",
    "how do i unlock this? what is the pin number?"
]
mountain = [
    "in what mountain was the ring forged",
    "a volcano in Congo erupted for the first time in nearly two decades, what was it's name?",
    "which icelandic volcano caused an intense seismic crisis 16 August 2014",
    "in tolkien's world, what mountain did Smaug the dragon inhabit",
    "what is the largest mountain chain",
    "what is the smallest mountain range in europe",
    "what is the tallest mountain in iberia",
    "name a mountain that is in the ocean",
    "Which is the world’s longest mountain system",
    "Name a volcano on the island of Hawaii",
    "Which volcano has the world’s largest rise from base to peak",
    "Name the world’s highest mountain",
    "Examples of Fold mountains",
    "Write some examples of fault-block mountains"
]
veh = [
    "what spaceship boldly went where no man had gone before",
    "what truck design made recent headlines",
    "what machine was 'Catbus' from the 1988 film 'My Neighbor Totoro'",
    "what vehicle is used to transport cargo to the ISS",
    "what is the name of the mars rover",
    "name 10 fictional vehicles",
    "Which Mercedes-Benz concept car featured a joystick instead of a steering wheel",
    "What British Car is held together by superglue",
    "what is the yellow thing in 'Yellow Submarine'",
    "what was the nuclear-powered bus from the 1976 film The Big Bus called",
    "give some examples of fictional aircraft",
    "List 20 fictional cars",
    "What is the official state car of the Emperor of Japan",
    "What’s the most stolen car in America",
    "What was the first car launched into space",
    "What was the first car to be mass-produced",
    "What was the first Japanese car to be produced in the United States",
    "What kind of car, inspired Ferruccio Lamborghini to start his own automobile company",
    "List the names of 4 fictional ships",
    "what fictional spacecraft do you know about",
    "what kind of vehicle was 'Priscilla' in the 1994 film 'The Adventures of Priscilla, Queen of the Desert'"
]
dist = [
    "The distance between the tops of two trees 20 m and 28 m high is 17 m. What is the horizontal distance between the two trees",
    "A car leaves London at 8.30am and arrives in Edinburgh at 5.30pm. If "
    "how far is it from London to Edinburgh",
    "A killer shark, attacking a fishing boat, swims at a speed of 13m/s for half a minute. How far does it swim in this time",
    "How many kilometers are in a mile",
    "How many miles are in a kilometer",
    "how many miles from Rome to Hispania",
    "what is the diameter of the observable universe",
    "what is the radius of uranus",
    "what is the distance from mars to the sun",
    "how far away is mars",
    "how far away is pluto",
    "how many astronomical units from the sun to the kuiper belt",
    "how many AU from the sun to venus",
    "what is the distance between the earth and the moon",
    "how far is Ceuta from Gibraltar",
    "how deep is mariana's trench",
    "how deep is the deepest hole in the world",
    "how tall is the tallest building",
    "what is the length of the eiffel tower",
    "what is the length of the tallest tree",
    "how many kilometers does the russian border span",
    "how many meters from point A to point B",
    "what is the diameter of the electron",
    "tell me the radius of the proton",
    "tell me the diameter of jupiter",
    "how high is mount everest",
    "how high is the smallest mountain",
    "how high is the tallest volcano",
    "how deep is the ocean",
    "A tree is broken by the wind. If the top of the tree struck the ground at an angle of 30° and at a distance of 30 m from the root, what is the height of the tree",
    "The angles of elevation of an artificial satellite measured from two earth stations are 30° and 40°, respectively. If the distance between the earth stations is 4000 km, what is the height of the satellite",
    "The length of the shadow of a vertical tower on level ground increases by 10 metres when the altitude of the sun changes from 45° to 30°. tell me the height of the tower ",
    "A 25 m ladder is placed against a vertical wall of a building. The ladder is 7 m from the base of the building. If the top of the ladder slips 4 m, how much will the foot of the ladder slide",
    "A horse runs for 2 hours 15 mins at a speed of 8 mph. How far does it run",
    "A car travels at a constant speed of 40 mph for 3 hours. How far does the car go",
    "How far did he drive down the road before he turned around and drove back if his trip took 5 hours"
]
ordnum = [
    "in the bible, What chapter has the most verses",
    "what place did you end up in the race",
    "what paragraph is the longest",
    "what is your place in line",
    "What ordinal number represents the position of Barack Obama in the succession of US presidents",
    "what position is 'filho da puta' in",
    "where are you in the leaderboard",
    "in what episode of the simpsons did that happen",
    "in which episode of star trek was that",
    "Where did Harvard fall on the U.S. News & World Report list this year",
    "what place did you end up in",
    "where did you rank in the olympics",
    "Which position were you in the queue",
    "What was your position in the queue",
    "What grade are you in",
    "On which floor is your apartment",
    "What child are you in your family? First, second, third?",
    "Are you the first, second or third child",
    "What is your place in the birth order of your family",
    "What is your place in the birth order of girls born to your parents",
    "Which is your precise position in the order of seniority",
    "Chronologically speaking, which child are you",
    "Where does the letter C come in alphabetical order",
    "Where did your horse come in the race",
    "What is your filial order of birth, to your parents",
    "I'm the nth child in my family. Which are you?",
    "What chronological position are you",
    "What is the ordinality of 6 in the set of even numbers?",
    "What is the ordinality of 11 in the set of odd numbers?",
    "Regarding the scoring order, which position do you hold",
    "The number two is the first even number. In the sequence of natural numbers, what is it's the position",
    "Where were you in the queue",
    "Sequentially, which prime is 5",
    "Sequentially, which place are you",
    "What is the rank of 7 in prime number series",
    "In a list of prime numbers, in which position does 5 appear",
    "In the list of your favorite animals, where are cats",
    "In the list of your favorite foods, where is fried chicken",
    "What ordinal number reflects the position of the number 3 in the set of prime numbers",
    "Among your sisters, where do you fall with respect to birth order",
    "Where did you come in the race",
    "What is the chronological number of Barack Obama as the President of United States",
    "What is the chronological position of Barack Obama as the President of United States",
    "What is the chronological position of Barack Obama among Presidents of the United States",
    "G. H. W. Bush was the 41st president, G. W. Bush was the 43rd president, but which one was J. F. Kennedy",
    "What's the chronological number of Joe Biden as a president of America",
    "What is Salazar's number in the order of portuguese rulers",
    "what place were you in the game",
    "Abraham Lincoln was which president, first?, second?, third?",
    "In which order Barack Obama become the president",
    "Where does Trump fall in the sequence of US presidents",
    "If George Washington was the 1st President of the United States, and John Adams was the 2nd, what number president is Barack Obama"
]
hum_title = [
    "what do you do for a living",
    "what is your line of work",
    "what is your job",
    "what do you call someone that invents things",
    "what is someone that writes software called",
    "can you tell me what your job is",
    "what is Daniel's profession",
    "state your occupation",
    "what is your dream job",
    "what is your profession",
    "what do you call your job",
    "what are you",  # an ugly motherfucker
    "what was einstein's job",
    "what do you call a person that sells books",
    "what do you call a person that works in the ISS",
    "what are the army highest ranking officers called",
    "what do you call someone that produces food",
    "what do you want to be when you grow up"
]
symbol = [
    "what is the periodic table symbol for Argon",
    "what symbol is used to represent the imaginary numbers",
    "what symbol represents the square root of negative one",
    "what is the symbol used by Sauron's armies",
    "what symbol identifies the uruk-hai of Saruman",
    "what tarot card symbolizes good fortune",
    "What is the chemical formula for LSD",
    "what signs are there in the horoscope",
    "what does the sign for 'no smoking' look like",
    "list 5 signs from the chinese zodiac",
    "what symbol is used to represent the american dollar",
    "what 16 additional signs did the Department of Transportation request in 1979",
    "identify the symbol for british pounds",
    "what are the most common Navigational symbols",
    "which certification marks signify conformance with a government or private organization's requirements",
    "what are printing registration marks intended for",
    "what sign means you can't turn right",
    "the silhouette of a broken wine glass is a shipping symbol, list 10 others",
    "what does the 'an umbrella with rain falling on it' symbol look like",
    "name all traffic signs",
    "what was the symbol used by the nazis",
    "The original set of DOT pictograms consisted of 34 symbols primarily intended for what",
    "the radiation symbol, separated from a box by a chevron, means what",
    "what are pictograms used to convey information useful to travelers without using words known as",
    "which one do you think is the most important traffic sign",
    "How well does this symbol represent the message",
    "What symbol do people commonly fail to understand the message that it denotes",
    "people from various cultures misunderstand this symbol, what is the symbol",
    "What symbol do you fail to understand",
    "What symbol do you think is the most difficult to learn",
    "Name a symbol that already been widely accepted",
    "What warning traffic signs usually take the shape of an equilateral triangle with a white background and thick red border",
    "name a symbol that seriously contradicts existing standards or conventions",
    "Give examples of symbols that contain elements that are unrelated to the message"
]
letter = [
    "what vowel do male names usually end with in portuguese",
    "what vowel do female names usually end with in spanish",
    "What is the alphabet for Persian",
    "name the least used consonant in the latin alphabet",
    "what is the first letter of the alphabet",
    "what is the first letter of the Greek alphabet"
]
# changes to original dataset
landmass = [
    "What peninsula is Spain part of ?",
    "Where is the Iberian peninsula ?",
    "What continent 's name appears on the upper left corner of a Budweiser label ?",
    "What continent is Argentina on ?",
    "What continent is Bolivia on ?",
    "Name 5 peninsulas",
    "in what peninsula is Portugal located",
    "What continent is Egypt on ?",
    "Which continent has the most roses ?",
    "What is the highest continent ?",
    "What continent pushes up the Executive Committee mountain range ?",
    "On what continent is Mozambique ?",
    "List all the continents",
    "Name all the continents",
    "What Will Earth's Next Supercontinent Be",
    "Which supercontinent existed at the time of the dinosaurs",
    "On which supercontinent did therapsids walk",
    "On which Hawaiian island is Pearl Harbor ?",
    "What 's the largest island in the West Indies ?",
    "Name 3 supercontinents",
    "What 's the second-largest island in the world ?",
    "What California bay 's largest island is Angel Island ?",
    "What Caribbean island is northeast of Trinidad ?",
    "What Caribbean island is sometimes called Little England ?",
    "What is the largest island in the Mediterranean Sea ?",
    "What Mediterranean island is home to the first Club Med ?",
    "What island group contains Jersey , Guernsey , Sark and Herm ?",
    "What island group is Guadalcanal a part of ?",
    "What island has a park called The Battery at is southern tip ?",
    "What island is home to statues called Mauis ?",
    "Which Archipelago is in the Aegean Sea",
    "What Archipelago does 'ilha das flores' belong to",
    "name the 2 Portuguese archipelagos",
    "what supercontinent existed during the late Paleozoic and early Mesozoic eras",
    "List 3 archipelagos along the coast of the Americas",
    "Name some archipelagos along the coast of Africa",
    "List 10 lakes or rivers in Iberia",
    "the continent we now know as North America was continuous with Africa, South America, and Europe. They all existed as a single continent named what",
    "what archipelagos can you find in the Caspian sea",
    "in what continent are the alps located",
    "What island was the target of the U.S. 's Operation Urgent Fury ?",
    "What islands got their name from the Spanish baja mar , meaning shallow water ?",

]
water = [
    "Which one of the Great Lakes is entirely within U.S. territory ?",
    "What strait links the Mediterranean Sea and the Atlantic Ocean ?",
    "What ocean did the Titanic sink in ?",
    "Where is the Orinoco River ?",
    "What is the location of Lake Champlain ?",
    "What is the largest lake in North America ?",
    "What ocean does Mauritania border ?",
    "What ocean is the largest in the world ?",
    "What ocean surrounds the Madeira Islands ?",
    "What ocean surrounds the Maldive Islands ?",
    "What ocean was Amelia Earhart flying over when she disappeared ?",
    "What river runs through Rowe , Italy ?",
    "On what river is Rome built ?",
    "On what river is Strasbourg built ?",
    "What 's the longest river in Canada ?",
    "What 's the longest river in the world ?",
    "What 's the sacred river of India ?",
    "What are the seven seas ?",
    "What is the location of the Sea of Tranquility ?",
    "What is the longest river in the United States ?",
    "What is the principal river of Ireland ?",
    "What are the world 's four oceans ?",
    "What body of water are the Canary Islands in ?",
    "What are all the rivers in Europe ?",
    "What are Britain 's two longest rivers ?",
    "What body of water does the Danube River flow into ?",
    "What body of water does the Yukon River empty into ?",
    "What are the world 's three largest oceans , in order of size ?",
    "What are the names of all the seas in the world and what ocean do they drain into ?",
    "What does the River Seine empty into ?",
    "What sea did the Romans call mare nostrum ?",
    "What sea is Bombay on ?",
    "What sea separates Naples and Algiers ?",
    "What sea surrounds the Cayman Islands ?",
    "What colorful sea 's region does Greek legend say the Amazons lived near ?",
    "What famed river flows through Bagdad ?",
    "What famed river was Hernando de Soto the first European to see ?",
    "What is the deepest area of the Arctic Ocean ?",
    "What is the deepest lake in the US ?",
    "What river does the Grand Coulee Dam dam ?",
    "What river flows between Fargo , North Dakota and Moorhead , Minnesota ?",
    "What river flows past the Temple of Karnak ?",
    "What river flows through Vienna , Budapest and Belgrade ?",
    "What river in the US is known as the Big Muddy ?",
    "What river is Pocahontas buried along ?",
    "What river is Windsor Castle on ?",
    "What river runs through Colorado , Kansas , and Oklahoma ?",
    "What river runs through Liverpool ?",
    "What lake in Scotland is said to hold one or more monsters ?",
    "What lake is Sheboygan on ?",
    "What is the location of Lake Champlain ?",
    "What is the largest natural lake in Pennsylvania ?",
    "What lake is the source of the White Nile ?",
]
landmass = [
    " ".join([w for w in q.rstrip(" ?").replace('"', "").replace(": ", "")
             .replace(" ,", ",").replace(" 's", "'s").replace("`", "'")
             .split(" ") if w])
    for q in landmass]
water = [
    " ".join([w for w in q.rstrip(" ?").replace('"', "").replace(": ", "")
             .replace(" ,", ",").replace(" 's", "'s").replace("`", "'")
             .split(" ") if w])
    for q in water]

expanded = questions + \
           ["LOC:water " + q for q in water] + \
           ["LOC:landmass " + q for q in landmass] + \
           ["LOC:mount " + q for q in mountain] + \
           ["ENTY:body " + q for q in body_part] + \
           ["ENTY:animal " + q for q in animal] + \
           ["ENTY:food " + q for q in food] + \
           ["ENTY:religion " + q for q in religion] + \
           ["NUM:temp " + q for q in temp] + \
           ["NUM:volsize " + q for q in vol] + \
           ["NUM:speed " + q for q in speed] + \
           ["NUM:weight " + q for q in weight] + \
           ["ENTY:color " + q for q in color] + \
           ["NUM:code " + q for q in num_code] + \
           ["NUM:dist " + q for q in dist] + \
           ["NUM:ord " + q for q in ordnum] + \
           ["ENTY:veh " + q for q in veh] + \
           ["HUM:title " + q for q in hum_title] + \
           ["ENTY:symbol " + q for q in symbol] + \
           ["ENTY:letter " + q for q in letter]

# cleanup and save
for q in water + landmass:
    if "LOC:other " + q in expanded:
        expanded.remove("LOC:other " + q)

expanded = [
    " ".join([w for w in q.rstrip(" ?").replace('"', "").replace(": ", "")
             .replace(" ,", ",").replace(" 's", "'s").replace("`", "'")
             .split(" ") if w])
    for q in expanded]

expanded = sorted(expanded)
pprint(expanded)

with open(join("clean_data", "raw_questions+.txt"), "w") as f:
    f.write("\n".join(expanded))
