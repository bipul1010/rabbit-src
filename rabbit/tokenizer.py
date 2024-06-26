from pathlib import Path
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: str) -> None:
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)

    @property
    def n_words(self):
        return self._model.vocab_size()

    @property
    def bos_id(self):
        return self._model.bos_id()

    @property
    def eos_id(self):
        return self._model.eos_id()

    @property
    def pad_id(self):
        return self._model.eos_id()

    def encode(self, s: str, bos: bool = False):
        t = self._model.Encode(s)
        if bos is True:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: list):
        return self._model.Decode(t)


if __name__ == "__main__":
    path = Path("./tokenizer/tokenizer.model")
    # path = Path("./rabbit/misc/tokenizer/tokenizer/tokenizer.model")
    tokenizer = Tokenizer(model_path=str(path))
    print(tokenizer.n_words)
    print(tokenizer.n_words, tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)

    text = f""""Market Forces" is the fourth episode of the animated television series The Spectacular Spider-Man, which is based on the comic book character Spider-Man, created by Stan Lee and Steve Ditko. In the episode, Spider-Man is hunted by Shocker, whose suit allows him to fire intense sonic blasts.

"Market Forces" was written by Andrew Robinson and directed by Dan Fausett. It incorporated computer-generated imagery in the sonic blasts used by Shocker, which mixed in with the other, traditional animation style used in the show. Shocker's secret identity was completely changed from that of his original comic book appearance, but his design stayed close to the original costume used.

The episode originally broadcast on March 22, 2008, on the Kids WB! block for the CW Network. It received generally positive reviews from television critics. IGN praised it for its imagery and storytelling, while iF Magazine said "Even the Shocker was more interesting on this show, so I continue to have high hopes for future episodes, characters, and villains."

Plot summary
Montana and The Enforcers steal a powered suit from an armored Tri-Corp truck after gassing the guards. Montana hands it to Hammerhead, who insists that Montana dons the suit to complete the "Big Man"'s contract to kill Spider-Man. The next day, Peter Parker gets across town as Spider-Man, unaware that Montana and his men are watching him. He then hangs out at Harry's apartment, discussing the upcoming Fall Formal, until he receives an e-mail from J. Jonah Jameson of the Daily Bugle informing Peter that Jonah wants to purchase his photos of Spider-Man. He leaves and promises to help Harry with homework later. At the Bugle building, Jonah mistakenly kicks Peter out before realizing who he is. Jonah pays him and makes him exit the building.

While heading back, Peter hears an alarm coming from a landfill and investigates. It ends up being a trap and he is attacked with sonic blasts by Montana, now wearing the suit and calling himself "Shocker". When he is close to moving in for the kill, one of the thugs used as bait, Alex O'Hirn, accidentally gives Spider-Man time to recover. Shocker then knocks him into a machine and, satisfied, leaves via helicopter. Spider-Man, however, survives but his paycheck was torn to shreds. The next day after school, where Harry is outraged with him over missing out on studying, Peter goes to replace his check at the Bugle where assistant editor Joe Robertson suggests getting a better camera. After Jonah takes a photo of Spider-Man covered in garbage as Peter's submission, Peter goes after O'Hirn and his partner Flint Marko as Spider-Man. He defeats them and tells them to inform Shocker he wants a rematch.

Peter makes it home in time for his curfew and spots Aunt May struggling with the bills, but must use the money he has to buy a new camera. When he goes to school the next day, he finds Harry is furious with him once more over forgetting their studying arrangements once more. At night, Hammerhead tells Shocker the Big Man is displeased with his failure. Meanwhile, Peter unsuccessfully tries to ask out Jonah's assistant, Betty Brant. After a tremor rattles the entire city, Peter, as Spider-Man, discovers it is the Shocker, leaving him a trail that leads to a condemned theater. During their fight, Spider-Man unsuccessfully tries to find out who hired him before finally bringing the building down and defeating the Shocker.

Meanwhile, Harry returns home where his father Norman tells him to take responsibility and study by himself. Norman then goes to meet with Hammerhead, revealing that he helped them steal the suit from Tri-Corp as they are his company's competitor. He talks over speaker phone with the Big Man, who wants him to create new supervillains to occupy Spider-Man in return for ample funding. At his house, Peter sends his photos to the Bugle and tries to give Aunt May the money, but she insists that he uses 10% of every paycheck to save for a new camera.

Production
"Market Forces" was written by Andrew Robinson and directed by Dan Fausett.[2][4] Though the show is done in the style of traditional animation, computer-generated imagery was used to produce the green sonic beams made by Shocker.[1] In the original comic book publications, Shocker's secret identity was a man named Herman Schultz. For The Spectacular Spider-Man, they changed his identity to that of Enforcer Montana, who had, in the comics, been a prominent character already.[1] Using Montana allowed the writers to not have to come up with a completely new origin for Shocker.[2] His suit was generally the same as that done in the comic books, but had extra features including goggles and vibrators.[1] His voice was provided by veteran voice performer Jeff Bennett.[4]

Cultural references
Greg Weisman, one of the producers for the series,[5] came up with the idea to do a title scheme for each arc. For "Market Forces" and its arc, the scheme is economics.[6] A line in the episode asks how deep a location is, using the Mariana Trench and the Ninth Circle of Hell in Dante's Inferno as comparisons.

Reception
"Market Forces" originally broadcast on March 22, 2008, on the Kids WB! block for the CW Network, at 10:00 a.m.[2][4] It received generally positive reviews from television critics. Eric Goldman of IGN gave the episode an 8.5 out of 10 and wrote "Sometimes change can be fun, and this episode was a great example of that." Goldman enjoyed the design of Shocker which he felt stayed true to his original design and was "intrigued" by changing his secret identity. He praised the usage of CGI, the portrayal of Peter's life, and the "cool" ending.[1]

Sean Elliot, the senior editor of iF Magazine, gave the episode an "A−" and said about the change of Shocker's secret identity, "saves the writers from having to come up with a completely different origin for a character that pretty much is a second tier villain anyways. " Elliot wrote that having Norman making a deal to produce new supervillains was "an extremely useful" convention that allows the introduction and creation of enemies for Spider-Man to fight.[2]
"""
    print(text)
    print(len(text.split()))

    t = tokenizer.encode(s=text)
    print(t, len(t))

    print(tokenizer.decode(t))
    # print(tokenizer.decode([tokenizer.bos_id] + t + [tokenizer.eos_id]))
