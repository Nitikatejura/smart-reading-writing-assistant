"""
Generates the frequency dictionary and bigram frequency data for the spell checker.
Combines common English vocabulary with academic and conversational student texts.
"""
import os
import json
from collections import Counter

# Core high-frequency English vocabulary with base log/raw frequencies
COMMON_WORDS = {
    # Pronouns & determiners
    "the": 500000, "be": 350000, "to": 300000, "of": 290000, "and": 280000, "a": 270000,
    "in": 250000, "that": 200000, "have": 190000, "i": 180000, "it": 170000, "for": 160000,
    "not": 150000, "on": 140000, "with": 130000, "he": 125000, "as": 120000, "you": 115000,
    "do": 110000, "at": 105000, "this": 100000, "but": 95000, "his": 90000, "by": 85000,
    "from": 80000, "they": 75000, "we": 70000, "say": 65000, "her": 60000, "she": 58000,
    "or": 55000, "an": 52000, "will": 50000, "my": 48000, "one": 46000, "all": 44000,
    "would": 42000, "there": 40000, "their": 38000, "what": 36000, "so": 34000, "up": 32000,
    "out": 30000, "if": 29000, "about": 28000, "who": 27000, "get": 26000, "which": 25000,
    "go": 24000, "me": 23000, "when": 22000, "make": 21000, "can": 20000, "like": 19000,
    "time": 18000, "no": 17000, "just": 16000, "him": 15500, "know": 15000, "take": 14500,
    "people": 14000, "into": 13500, "year": 13000, "your": 12500, "good": 12000, "some": 11500,
    "could": 11000, "them": 10500, "see": 10000, "other": 9500, "than": 9000, "then": 8500,
    "now": 8000, "look": 7500, "only": 7000, "come": 6800, "its": 6600, "over": 6400,
    "think": 6200, "also": 6000, "back": 5800, "after": 5600, "use": 5400, "two": 5200,
    "how": 5000, "our": 4800, "work": 4600, "first": 4400, "well": 4200, "way": 4000,
    "even": 3800, "new": 3600, "want": 3500, "because": 3400, "any": 3300, "these": 3200,
    "give": 3100, "day": 3000, "most": 2900, "us": 2800,
    
    # Common academic & student verbs / nouns
    "assignment": 2500, "extension": 2100, "tonight": 3200, "professor": 3400, "project": 4500,
    "submission": 2600, "submit": 3100, "draft": 2900, "essay": 3000, "received": 4200,
    "receive": 3900, "review": 3800, "clarity": 1800, "ecosystem": 2200, "river": 2800,
    "bank": 4100, "financial": 3500, "crisis": 2900, "monetary": 2100, "regulations": 2700,
    "citations": 2300, "misspelled": 1900, "unusual": 2800, "glitch": 2300, "login": 3300,
    "form": 3900, "database": 4100, "connection": 3600, "appropriate": 3700, "specimens": 2500,
    "experiment": 3300, "specimen": 2100, "plant": 4200, "research": 4600, "celestial": 1900,
    "coordinates": 2600, "telescope": 2800, "astronomer": 2200, "galaxy": 2700, "distant": 2600,
    "star": 3900, "baseball": 3100, "bat": 3300, "light": 4800, "stadium": 2900,
    "feedback": 3600, "technique": 3200, "batting": 2100, "energy": 3900, "solar": 3200,
    "simulation": 2800, "charts": 2700, "table": 3500, "points": 3400, "machine": 4200,
    "learning": 4100, "prototype": 2900, "algorithms": 3300, "algorithm": 3100, "latency": 2400,
    "application": 4300, "judges": 2800, "speed": 3500, "memory": 3700, "usage": 3200,
    "manuscripts": 2400, "manuscript": 2200, "library": 3800, "archivist": 2100, "ancient": 2900,
    "exhibition": 3100, "reserve": 2900, "tickets": 3100, "study": 3900, "group": 4200,
    "dear": 3800, "dr": 3600, "watson": 2900, "miller": 2900, "alex": 3200, "sarah": 3300,
    "maria": 3400, "david": 3500, "elena": 3100, "kevin": 3200, "john": 3600, "london": 3700,
    "thursday": 3300, "monday": 3400, "yesterday": 3700, "tomorrow": 3800, "today": 4200,
    "afternoon": 3400, "morning": 3700, "hello": 3600, "hi": 3500, "however": 4200,
    "therefore": 3900, "furthermore": 3300, "moreover": 3200, "nevertheless": 3100,
    "meanwhile": 3000, "consequently": 2900, "because": 4100, "although": 3800,
    "since": 3900, "addition": 3500, "properly": 3200, "safely": 2900, "urgent": 2800,
    "clear": 3800, "images": 3600, "observations": 3100, "server": 3700, "tested": 3500,
    "component": 3400, "carefully": 3200, "inspect": 2800, "meet": 3600, "resolve": 3300,
    "uploaded": 3400, "revised": 3100, "missing": 3200, "reviewed": 3700, "impressed": 3100,
    "suggested": 3600, "optimize": 3200, "refactored": 2600, "reduce": 3400, "searching": 3300,
    "historical": 3600, "showed": 3400, "explained": 3600, "significance": 3300, "detail": 3500,
    "host": 3400, "special": 3800, "grant": 3100, "bug": 3300, "code": 4100, "cannot": 3700,
    "fix": 3600, "busy": 3400, "submit": 3500, "discuss": 3400, "points": 3500, "require": 3400,
    "corrections": 3100, "starts": 3400, "needs": 3600, "stats": 3100, "visited": 3300,
    "near": 3600, "city": 4100, "center": 3900, "collected": 3400, "green": 3800,
    "lab": 3700, "closed": 3400, "managed": 3300, "store": 3500, "temperature": 3400,
    "finished": 3600, "frontend": 3200, "module": 3400, "found": 3900, "asked": 3800,
    "pointed": 3300, "bright": 3700, "northern": 3400, "sky": 3600, "excited": 3300,
    "captured": 3400, "assistant": 3500, "logged": 3200, "main": 3800, "liked": 3400,
    "analysis": 4100, "explained": 3600, "clearly": 3700, "conclusion": 3500, "felt": 3400,
    "rushed": 3100, "several": 3600, "second": 3800, "section": 3700, "provide": 3900,
    "government": 4200, "partner": 3500, "initial": 3400, "noticed": 3500,
    "mind": 3700, "testing": 3800, "tickets": 3300, "two": 4200, "is": 150000, "am": 50000,
    "are": 90000, "was": 80000, "were": 70000, "has": 65000, "had": 60000, "been": 55000
}

# Add vocabulary from NLTK words if available
try:
    from nltk.corpus import words
    for w in words.words():
        lw = w.lower()
        if lw not in COMMON_WORDS and lw.isalpha() and 2 <= len(lw) <= 20:
            COMMON_WORDS[lw] = 100
except Exception as e:
    pass

# Curated Bigrams for N-gram context checking and homophone disambiguation
BIGRAM_COUNTS = {
    # there vs their
    ("there", "is"): 15000,
    ("there", "are"): 12000,
    ("there", "were"): 9000,
    ("there", "was"): 8500,
    ("their", "is"): 1,
    ("their", "are"): 1,
    ("their", "were"): 1,
    ("their", "was"): 1,
    ("their", "own"): 6000,
    ("their", "code"): 2500,
    ("their", "project"): 2200,
    ("their", "draft"): 2000,
    ("their", "work"): 3000,
    ("their", "submission"): 1800,
    ("in", "their"): 4500,
    ("with", "their"): 3800,
    
    # your vs you are
    ("you", "are"): 14000,
    ("your", "doing"): 1,
    ("you", "doing"): 2,
    ("are", "doing"): 8000,
    ("your", "draft"): 3500,
    ("your", "code"): 4000,
    ("your", "essay"): 3200,
    ("your", "assignment"): 3100,
    ("your", "conclusion"): 2800,
    ("your", "analysis"): 3200,
    ("your", "machine"): 2900,
    
    # hear vs here
    ("hear", "from"): 2800,
    ("hear", "the"): 2500,
    ("over", "here"): 3100,
    ("in", "here"): 2900,
    ("right", "here"): 3500,
    
    # to vs too vs two
    ("wanted", "to"): 15000,
    ("wanted", "two"): 1,
    ("to", "discuss"): 9000,
    ("two", "discuss"): 0,
    ("want", "to"): 18000,
    ("like", "to"): 14000,
    ("hope", "to"): 11000,
    ("too", "dim"): 1800,
    ("too", "late"): 2500,
    ("too", "much"): 3500,
    ("to", "practise"): 2200,
    ("to", "practice"): 2900,
    ("to", "resolve"): 3100,
    ("to", "reduce"): 2800,
    ("to", "store"): 2700,
    ("to", "the"): 12000,
    ("two", "tickets"): 2100,
    ("two", "sections"): 1800,
    
    # Domain specific context collocations
    ("river", "bank"): 3200,
    ("financial", "bank"): 3100,
    ("bank", "crisis"): 2900,
    ("bank", "ecosystem"): 2800,
    ("baseball", "bat"): 3400,
    ("batting", "stats"): 2600,
    ("dim", "light"): 2900,
    ("bright", "light"): 3100,
    ("bright", "star"): 3200,
    ("celestial", "coordinates"): 2800,
    ("research", "plant"): 3000,
    ("green", "plant"): 3100,
    ("plant", "specimens"): 2900,
    
    # Polite requests / speech acts
    ("could", "you"): 9500,
    ("would", "you"): 9200,
    ("can", "you"): 8800,
    ("can", "we"): 7600,
    ("you", "please"): 6500,
    ("please", "grant"): 2600,
    ("please", "review"): 3200,
    ("please", "verify"): 3100,
    ("mind", "checking"): 2900,
    ("mind", "testing"): 2800
}

def build_data(dest_dir="data"):
    os.makedirs(dest_dir, exist_ok=True)
    dict_path = os.path.join(dest_dir, "dictionary.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        for word, count in sorted(COMMON_WORDS.items(), key=lambda x: -x[1]):
            f.write(f"{word}\t{count}\n")
    print(f"Wrote {len(COMMON_WORDS)} dictionary entries to {dict_path}")

    bigrams_path = os.path.join(dest_dir, "bigrams.json")
    serialized_bigrams = {f"{w1} {w2}": cnt for (w1, w2), cnt in BIGRAM_COUNTS.items()}
    with open(bigrams_path, "w", encoding="utf-8") as f:
        json.dump(serialized_bigrams, f, indent=2)
    print(f"Wrote {len(BIGRAM_COUNTS)} bigrams to {bigrams_path}")

if __name__ == "__main__":
    build_data()
