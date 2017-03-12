file = open('yelp_academic_dataset_review.json')

# JJ: adjective or numeral, ordinal
# JJR: adjective, comparative
# JJS: adjective, superlative
# NN: noun, common, singular or mass
# NNP: noun, proper, singular
# NNPS: noun, proper, plural
# NNS: noun, common, plural
# PRP: pronoun, personal
# PRP$: pronoun, possessive
# RB: adverb
# RBR: adverb, comparative
# RBS: adverb, superlative
# VB: verb, base form
# VBD: verb, past tense
# VBG: verb, present participle or gerund
# VBN: verb, past participle
# VBP: verb, present tense, not 3rd person singular
# VBZ: verb, present tense, 3rd person singular

markers = {
"JJ" : 0,
"JJR" : 0,
"JJS" : 0,
"NN" : 1,
"NNP" : 1,
"NNPS" : 1,
"NNS" : 1,
"PRP" : 1,
"PRP$" : 1,
"RB" : 2,
"RBR" : 2,
"RBS" : 2,
"VB" : 3,
"VBD" : 3,
"VBG" : 3,
"VBN" : 3,
"VBP" : 3,
"VBZ" : 3,
}

import nltk
# Length, Verb, Adverb, Adjective, Noun
dataset = []
# count = 
lenVsRating = {}
i = 0
for l in file:
	line = eval(l)
	# print line 
	
	sentence = line['text']
	# print i
	i+= 1
	if i == 5000:
		break
	# print sentence
	# sentence = "I have multiple texts and I would like to create profiles of them based on their usage of various parts of speech, like nouns and verbs. Basially, I need to count how many times each part of speech is used"
	try:
		tokens = nltk.word_tokenize(sentence.lower())
		text = nltk.Text(tokens)
		tags = nltk.pos_tag(text)
		vec = [0]*6

		# for i in sentence:
		vec[0] = len(sentence)
		for t in tags:
			tag = t[1]
			if tag in markers.keys():
				# if tag[0] == 'V':
				# 	vec[1] += 1
				# if tag[0] == 'J':
				# 	vec[3] += 1
				if tag[0] == 'N':
					vec[4] += 1
				# if tag[0] == 'R':
				# 	vec[2] += 1
		vec[5] = float(line['stars'])
		dataset.append(vec)
		k = 3
		if(vec[4]/k not in lenVsRating):
			lenVsRating[vec[4]/k] = []
		lenVsRating[vec[4]/k].append(vec[5])
	except UnicodeDecodeError:
		print "Found error"
		continue

	# print nltk.help.upenn_tagset()
import numpy as np
import pylab as plt
nvec = np.array(dataset)



from scipy.stats.stats import pearsonr

# ratings = nvec[:, 5]
# lens = nvec[:, 0]
# args = np.argsort(lens)
# lens = lens[args]
# ratings = ratings[args]

# plt.plot(lens, ratings)

kys = lenVsRating.keys()
kys = sorted(kys)

x = []
y = []
for ky in kys:
	print ky, 
	print lenVsRating[ky]
	x.append(ky)
	y.append(np.mean(lenVsRating[ky]))


# plt.scatter(np.array(x)*k, y, s = 4)
# plt.show()

plt.scatter(np.array(x)*k, y, s = 150, color = 'navy', marker = '+')
plt.xlabel('Number of adjectives used')
plt.ylabel('Rating')
print pearsonr(x, y)
plt.show()


# Len - -0.6262
# Verb - 0.56
# Adjective - 0.25
# Noun - 0.047
