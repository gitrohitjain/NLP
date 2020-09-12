import nltk
import re
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer,WordNetLemmatizer
from gensim.models import Word2Vec

para = '''
Oh, it just feels like an incredible understatement to say how grateful I am to be here with all of you. I feel like I have a relationship with many of you on social media, and you were like, “T-minus two days.” I'm like, “It's coming! We're going to be together.” So I'm so grateful to be here with you.

I'm going to talk about trust and I'm going to start by saying this: One of my favorite parts of my job is that I get to research topics that mean something to me. One of my least favorite parts of my job is I normally come up with findings that kicked me in the butt and make me change my entire life. That's the hard part. But I get to dig into the stuff that I think matters in my life and the life of the people around me.

And the topic of trust is something I think I probably would have eventually started to look at closely because I study shame and vulnerability. But there's a very personal reason I jumped to trust early in my research career, and it was a personal experience.

One day, my daughter, Ellen, came home from school. She was in third grade. And the minute we closed the front door, she literally just started sobbing and slid down the door until she was just kind of a heap of crying on the floor. And of course I was … It scared me, and I said, “What's wrong Ellen? What happened? What happened?”

And she pulled herself together enough to say, “Something really hard happened to me today at school, and I shared it with a couple of my friends during recess. And by the time we got back into the classroom, everyone in my class knew what had happened, and they were laughing and pointing at me and calling me names.” And it was so bad, and the kids were being so disruptive, that her teacher even had to take marbles out of this marble jar.

And the marble jar in the classroom is a jar where if the kids are making great choices together, the teacher adds marbles. If they're making not great choices, the teacher takes out marbles. And if the jar gets filled up, there's a celebration for the class.

And so, she said, “It was one of the worst moments in my life. They were laughing and pointing. And Miss Bacchum, my teacher, kept saying, ‘I'm going to take marbles out.' And she didn't know what was happening.”

And she looked at me just with this face that is just seared my mind and said, “I will never trust anyone again.” And my first reaction, to be really honest with you, was, “Damn straight, you don't tell anybody anything but your Mama.”

Yeah, right? That's it. I mean, that was my … “You just tell me. And when you grow up and you go off to school, Mama will go too. I'll get a little apartment.” And the other thing I was thinking to be quite honest with you is, “I will find out who those kids were.” And while I'm not going to beat up a nine year old, I know their mamas.

You know, that's the place you go to. And I'm like, “How am I going to explain trust to this third grader in front of me?” So I took a deep breath and I said, “Ellen, trust is like a marble jar.” She said, “What do you mean?” And I said, “You share those hard stories and those hard things that are happening to you with friends, who, over time, you filled up their marble jar. They've done thing after thing after thing where you're like, ‘I know I can share this with this person.' Does that make sense?”

Yes!

And that's what Ellen said, “Yes, that makes sense.” And I said, “Do you have any marble jar friends?” And she said, “Oh yeah. Totally. Hannah and Lorna are marble jar friends.” And I said … And then this is where things got interesting. I said, “Tell me what you mean. How do they earn marbles for you?”

And she's like, “Well, Lorna, if there's not a seat for me at the lunch cafeteria, she'll scoot over and give me half a heinie seat.” And I'm like, “She will?” She's like, “Yeah. She'll just sit like that, and so I can sit with her.” And I said, “That's a big deal.” This is not what I was expecting to hear.

And then she said, “And you know Hannah, on Sunday at my soccer game?” And I was waiting for this story where she said, “I got hit by a ball and I was laying on the field, and Hannah picked me up and ran me to first aid.” And I was like, “Yeah?” And she said, “Hannah looked over and she saw Oma and Opa,” my parents, her grandparents, “And she said, ‘Look, your Oma and Opa are here.'” And I was like …

And I was like, “Boy, she got a marble for that?” And she goes, “Well, you know, not all my friends have eight grandparents.” Because my parents are divorced and remarried, my husband's parents were divorced and remarried. And she said, “And it was so nice to me that she remembered their names.”

And I was like, “Hmm.” And she said, “Do you have marble jar friends?” And I said, “Yeah, I do have a couple of marble jar friends.” And she said, “Well, what kind of things do they do to get marbles?” And this feeling came over me. And I thought … The first thing I could think of, because we were talking about the soccer game, was that same game. My good friend Eileen walked up to my parents and said, “Diane, David, good to see you.” And I remember what that felt like for me. And I was like, certainly, trust cannot be built by these small insignificant moments in our lives. It's gotta be a grander gesture than that.

So, as a researcher, I start looking into the data. I gather up the doctoral students who've worked with me. We start looking. And it is crystal clear. Trust is built in very small moments. And when we started looking at examples of when people talked about trust in the research, they said things like, “Yeah, I really trust my boss. She even asked me how my mom's chemotherapy was going.” “I trust my neighbor because if something's going on with my kid, it doesn't matter what she's doing, she'll come over and help me figure it out.” You know, one of the number one things emerged around trust and small things? People who attend funerals. “This is someone who showed up at my sister's funeral.”

Another huge marble jar moment for people, “I trust him because he'll ask for help when he needs it.” How many of you are better at giving help than asking for help? Right? So, asking for help is one of those moments.

So, one of the ways I work as a grounded theory researcher, is I look at the data first, then I go in and see what other researchers are talking about and saying, because we believe the best theories are not built on other existing theories, but on our own lived experiences.

So, after I had looked at this, I said, “Let me see what the research says.” And I went to John Gottman, who's been studying relationship for 30 years. He has amazing work on trust and betrayal. And the first thing I read, “Trust is built in the smallest of moments.” And he calls them “Sliding door moments.”

Sliding Doors is a movie with Gwyneth Paltrow from the 90s. Have you all seen this movie? So, it's a really tough movie, because what happens is it follows her life to this seemingly unimportant moment where she's trying to get on a train. And she makes the train, but the movie stops and splits into two parts where she makes a train and she doesn't make the train, and it follows them to radically different endings. And he would argue that trust is a sliding door moment. And the example that he gives is so powerful.

He said he was lying in bed one night, he had 10 pages left of his murder mystery, and he had us feeling he knew who the killer was, but he was dying to finish this book. So he said, “I don't even want … I want to get up, brush my teeth, go to the bathroom, and get back in and not have to get up.” You know that feeling when you just want to get all situated and read the end of your book?

So, he gets up and he walks past his wife in the bathroom, who's brushing her hair and who looks really sad. And he said, “My first thought was just keep walking. Just keep walking.”

And how many of you have had that moment you walk past someone and you're like, “Oh, God. They look … Avert your eyes.” Or you look at caller ID or your cell phone, and you're like, “Oh yeah, I know she's in a big mess right now. I don't have time to pick up the phone.” Right? Yes or no? This looks like guilty laughter to me.

So, he said, “That's a sliding door moment.” And here's what struck me about his story, because he said, “There is the opportunity to build trust and there is the opportunity to betray.” Because as small as the moments of trust can be, those are the moments of betrayal as well. To choose to not connect when the opportunity is there is a betrayal. So he took the brush out of her hand and started brushing her hair and said, “What's going on with you right now, babe?” That's a moment of trust, right?

So fast-forward five years, and I'm clear about trust, and I talk about trust as the marble jar. We've got to really share our stories and our hard stuff with people whose jars are full, people who've, over time, really done those small things that have helped us believe that they're worth our story.

But the new question for me was this: What are those marbles? What is trust? What do we talk about when we talk about trust? Trust is a big word, right? To hear, “I trust you,” or “I don't trust you.” I don't even know what that means. So, I wanted to know, what is the anatomy of trust? What does that mean?

So, I started looking in the research and I found a definition from Charles Feldman that I think is the most beautiful definition I've ever heard. And it's simply this: “Trust is choosing to make something important to you vulnerable to the actions of someone else.” “Choosing to make something important to you vulnerable to the actions of someone else.” Feldman says that distrust is what I have shared with you that is important to me is not safe with you.

So, I thought, “That's true.” And Feldman really calls for this, let's understand what trust is. So, we went back into all the data to find out, can I figure out what trust is? Do I know what trust is from the data? And I think I do know what trust is.

And I put together an acronym, BRAVING, B-R-A-V-I-N-G. BRAVING. Because when we trust, we are braving connection with someone. So what are the parts of trust? B, boundaries. I trust you. If you are about your boundaries and you hold them, and you're clear about my boundaries and you respect them. There is no trust without boundaries.

R, reliability. I can only trust you if you do what you say you're going to do. And not once. Reliability … Let me tell you what reliability is in research terms. We're always looking for things that are valid and reliable. Any researchers here or research kind of geeks? There's 10 of us.

Okay. So we would say a scale that you weigh yourself on is valid if you get on it and it's an accurate weight. 120. Okay. So that would be a very valid scale. I would pay a lot of money for that scale. So, that's actually not a valid scale, but we'll pretend for the sake of this. That's a valid scale.

A reliable scale is a scale that if I got on it a hundred times, it's gonna say the same thing every time. So, what reliability is, is you do what you say you're going to do over and over and over again. You cannot gain and earn my trust if you're reliable once, because that's not the definition of reliability.
'''


text = re.sub(r'\[[0-9]*\]',' ',para)
text = re.sub(r'\s+',' ',text) #whitespace character
text = text.lower()
text = re.sub(r'\d',' ',text) #decimal digit
text = re.sub(r'\s+',' ',text) 


sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['wife']

# Most similar words
similar = model.wv.most_similar('doctoral')
