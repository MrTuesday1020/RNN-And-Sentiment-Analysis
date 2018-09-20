import re

fread = open('link_words.txt', "r")

words = {*()};

line = fread.readline()
while line:
	line = line.lower()
	line = re.sub(r"\n", "", line)
	words.add(line)
	line = fread.readline()

print(words)
fread.close()