"""
Generate LARGE Likert Scale Training Dataset
Creates 3000 labeled examples (600 per class) for better model accuracy
"""

import pandas as pd
import random

random.seed(42)

def generate_large_dataset():
    """Generate 3000 labeled examples across 5 classes"""
    
    data = []
    
    # SCORE 5: Strong Agreement (600 examples)
    strong_agree_base = [
        "I completely agree", "Absolutely correct", "I couldn't agree more", 
        "This is exactly right", "100% yes", "You're absolutely right",
        "I'm fully in support", "This perfectly describes my view", "Exactly my thoughts",
        "I strongly believe this", "Without a doubt, I agree", "This is definitely the way",
        "I'm all for this", "Perfectly said", "Yes, I'm completely behind this",
        "This resonates completely", "Absolutely correct in every way", "I enthusiastically support",
        "This is precisely how I feel", "Couldn't have said it better", "I love this idea",
        "This is exactly right", "I'm totally convinced", "Yes! This is what I believe",
        "I wholeheartedly agree", "This aligns perfectly", "Absolutely, no question",
        "I'm in complete agreement", "This is undeniably true", "I support this 100%",
        "Definitely agree completely", "This is brilliant", "Yes, without hesitation",
        "I'm on the same page", "This captures my opinion", "Absolutely yes, strongly agree",
        "I find this convincing", "This is perfectly accurate", "I'm totally in favor",
        "Yes, makes complete sense", "I agree wholeheartedly", "Excellent point",
        "I'm fully aligned", "This is absolutely right", "Yes, completely convinced",
        "This is how it should be", "Total agreement", "Absolutely, couldn't agree more",
        "This is my stance", "Fully committed", "Yes, this is valid",
        "I agree with every aspect", "Fundamentally correct", "100% convinced",
        "Absolutely true", "I enthusiastically endorse", "Yes, I strongly believe",
        "Perfectly reflects my opinion", "Entirely in agreement", "This is the case",
        "I fully embrace this", "Yes, without doubt", "Completely aligned",
        "Unquestionably correct", "I agree 100%", "No reservations",
        "Totally on board", "Exactly what's needed", "Couldn't agree more",
        "Yes, completely true", "Full support", "This is what I think",
        "I agree completely", "Thoroughly convinced", "Yes, spot on",
        "I agree every way", "This is my position", "Completely supportive",
        "Undeniably true", "I agree without question", "Perfectly stated",
        "Yes, firmly agree", "I enthusiastically agree", "This is my view",
        "Completely convinced", "Yes, with confidence", "Entirely correct",
        "Totally committed", "I agree unreservedly", "Completely persuasive",
        "Yes, fundamentally true", "In complete accord", "This is my belief",
        "I agree unequivocally", "Absolutely yes", "Entirely convinced",
        "Yes, precisely right", "Unreservedly agree", "Perfectly accurate",
        "Of course I agree", "Indeed, I concur", "Most certainly",
        "Exactly correct", "I'm with you 100%", "Totally right",
        "Absolutely spot on", "I'm in full agreement", "This is perfect",
        "Yes definitely", "I strongly agree", "This is ideal",
        "I'm completely for it", "Yes absolutely", "I fully support",
        "This is wonderful", "I'm all in", "Yes wholeheartedly",
        "I totally back this", "This is excellent", "I'm entirely behind it",
        "Yes with enthusiasm", "I'm fully committed", "This is outstanding",
        "I'm 100% in favor", "Yes without question", "I'm totally supportive",
        "This is superb", "I'm completely onboard", "Yes I'm convinced",
        "I'm fully persuaded", "This is brilliant", "I'm entirely in favor",
        "Yes I'm certain", "I'm totally behind it", "This is fantastic",
        "I'm completely sold", "Yes I'm sure", "I'm fully in accord",
        "This is great", "I'm entirely supportive", "Yes I believe it",
        "I'm completely positive", "This is terrific", "I'm fully onboard",
        "Yes I think so", "I'm totally in agreement", "This is amazing",
        "I'm completely satisfied", "Yes I'm positive", "I'm fully behind it",
        "This is perfect sense", "I'm entirely convinced", "Yes for sure",
        "I'm completely in favor", "This is exactly what I think", "I'm fully agreeable",
    ]
    
    # SCORE 4: Agreement (600 examples)
    agree_base = [
        "I generally agree", "I think this is mostly correct", "I agree for the most part",
        "This seems reasonable", "I lean towards agreeing", "I'm inclined to agree",
        "There's merit to this", "I agree with most of it", "Makes sense overall",
        "I'm somewhat in agreement", "I tend to agree", "This is fairly accurate",
        "I agree more than disagree", "Sounds about right", "I'm mostly on board",
        "This is largely true", "I agree with main points", "Probably correct",
        "Leaning towards yes", "This is generally valid", "I agree overall",
        "This seems like a good point", "I'm fairly convinced", "This holds truth",
        "I agree on balance", "Makes a lot of sense", "Pretty much agree",
        "This is reasonable", "I agree with the idea", "Fairly accurate",
        "I'm mostly convinced", "Largely correct", "I agree with premise",
        "Sounds plausible", "Inclined to think right", "I agree with much",
        "Solid point", "Leaning in favor", "Mostly true",
        "I agree with core", "Makes considerable sense", "Generally supportive",
        "Fairly valid", "I agree with basics", "Reasonable overall",
        "Somewhat convinced", "There's truth here", "I agree in principle",
        "Pretty accurate", "Fairly in agreement", "Mostly sound",
        "I agree with thrust", "Fair point", "Leaning towards agreement",
        "Generally right", "I agree with aspects", "Makes good sense",
        "Largely in agreement", "Quite valid", "I agree with essence",
        "Mostly correct", "Pretty convinced", "On right track",
        "I agree with fundamentals", "Reasonable enough", "Generally in favor",
        "This has merit", "I agree broadly", "Fairly sound",
        "Somewhat supportive", "Largely sensible", "I agree with concept",
        "Makes sense mostly", "Fairly persuaded", "Reasonably accurate",
        "I agree with key points", "Decent argument", "Leaning positive",
        "Generally good", "I agree with message", "Fairly convincing",
        "Mostly same page", "Pretty reasonable", "I agree with central idea",
        "Largely valid", "Fairly aligned", "Mostly acceptable",
        "I agree with direction", "Fair amount of sense", "Somewhat persuaded",
        "Quite plausible", "I agree with basic point", "Reasonable to me",
        "Generally convinced", "Fairly solid", "I agree primarily",
        "Sounds mostly right", "I think so", "Yeah I'd say so",
        "Pretty much", "I suppose so", "Seems right",
        "I'd agree", "Makes sense to me", "I can go along with that",
        "Fair enough", "I see the point", "That works",
        "I'm okay with that", "Sounds good", "I can accept that",
        "That's reasonable", "I'm fine with it", "Makes sense",
        "I can agree to that", "Seems fair", "I'll go with that",
        "That's acceptable", "I'm comfortable with it", "Sounds reasonable",
        "I can support that", "That's fine", "I'm agreeable",
        "Seems sensible", "I can back that", "That's okay",
        "I'm on board with it", "Sounds fair", "I can agree",
        "That's plausible", "I'm supportive", "Seems valid",
        "I can endorse that", "That's sound", "I'm favorable",
        "Seems correct", "I can approve that", "That's good",
        "I'm positive about it", "Sounds accurate", "I can confirm that",
        "That's right", "I'm inclined to agree", "Seems true",
        "I can validate that", "That's proper", "I'm disposed to agree",
    ]
    
    # SCORE 3: Neutral (600 examples)
    neutral_base = [
        "I'm not sure", "I have mixed feelings", "I can see both sides",
        "I'm undecided", "Neither agree nor disagree", "I'm on the fence",
        "No strong opinion", "I'm neutral", "It depends",
        "I see merit on both sides", "I'm ambivalent", "Need more information",
        "I'm in the middle", "I have reservations", "Not convinced either way",
        "It's complicated", "I understand different perspectives", "I'm indifferent",
        "No particular view", "I see pros and cons", "I'm impartial",
        "Uncertain about stance", "It's debatable", "Haven't formed opinion",
        "I see validity in multiple views", "Not leaning either way", "There are trade-offs",
        "Unsure where I stand", "Arguments on both sides", "I'm noncommittal",
        "It's complex", "Balanced in view", "Don't have clear position",
        "See reasoning but concerns", "Middle-of-the-road", "People can disagree",
        "Open to interpretations", "Not pulled either way", "I see nuance",
        "I'm equivocal", "Context matters", "Don't feel strongly",
        "I see the dilemma", "In middle ground", "No clear answer",
        "Competing thoughts", "Multiple angles", "Unsure of position",
        "It's subjective", "I'm torn", "No definitive view",
        "See advantages and disadvantages", "Uncertain", "Opinions vary",
        "Not committed either side", "I see complexity", "I'm indeterminate",
        "Different considerations", "Neutral assessment", "Haven't decided",
        "See strengths and weaknesses", "In between", "Not straightforward",
        "Open-minded", "Don't take firm stance", "Multiple perspectives",
        "Balanced thinking", "Debatable either way", "Not inclined strongly",
        "Gray areas", "Undetermined", "Circumstances matter",
        "In neutral zone", "Haven't formed judgment", "Arguments both ways",
        "Unresolved", "Open to interpretation", "On sidelines",
        "Don't lean one way", "Challenges in deciding", "Middle-ground",
        "Minds differ", "Not fixed in view", "Conflicting considerations",
        "Impartial judgment", "Mixed bag", "Not decided",
        "Valid points everywhere", "Neutral territory", "No simple answer",
        "Open to persuasion", "No strong feeling", "Tension between views",
        "Uncommitted", "Depends on factors", "Could go either way",
        "Sitting this out", "Hard to say", "Maybe",
        "Perhaps", "Possibly", "Not certain",
        "Can't say", "Unknown", "Unclear",
        "Debatable", "Questionable", "Unsure",
        "Don't know", "No idea", "Can't tell",
        "Uncertain", "Up in the air", "TBD",
        "We'll see", "Time will tell", "Remains to be seen",
        "Not sure yet", "Still thinking", "Haven't made up my mind",
        "On the fence about it", "Fifty-fifty", "Equal weight",
        "Balanced view", "Middle position", "Central stance",
        "Moderate opinion", "Temperate view", "Even-handed",
        "Fair-minded", "Unprejudiced", "Objective",
        "Dispassionate", "Detached", "Unbiased",
        "Neutral stance", "Middle way", "Centrist view",
    ]
    
    # SCORE 2: Disagreement (600 examples)
    disagree_base = [
        "I'm not really convinced", "I have some doubts", "I lean towards disagreeing",
        "I'm somewhat skeptical", "Don't think this is right", "I have reservations",
        "I'm inclined to disagree", "Not sure I agree", "There are issues",
        "I disagree to some extent", "Not fully on board", "This is questionable",
        "I have concerns", "Leaning towards no", "Don't think accurate",
        "Somewhat opposed", "This overlooks points", "Not particularly convinced",
        "I disagree more than agree", "I have doubts", "Fairly skeptical",
        "Don't think this holds up", "Somewhat against", "This is problematic",
        "Not really in agreement", "I have objections", "Inclined to think wrong",
        "Don't find convincing", "This misses the mark", "Mostly unconvinced",
        "I disagree with much", "This is flawed", "Not supportive",
        "I have problems", "Leaning against", "Don't think makes sense",
        "Somewhat critical", "This is off-base", "Not buying this",
        "I disagree with main points", "This is weak", "Fairly opposed",
        "Issues with argument", "Not persuaded", "This is unsound",
        "Somewhat dissenting", "Don't think valid", "I have disagreements",
        "Leaning negative", "This is unconvincing", "Not really aligned",
        "I disagree with premise", "This is dubious", "Fairly critical",
        "Don't accept view", "Somewhat resistant", "This doesn't work",
        "Not in favor", "I disagree on points", "This is mistaken",
        "Mostly against", "Strong doubts", "Not convinced",
        "This is incorrect", "Somewhat dissatisfied", "Don't agree with approach",
        "Major reservations", "Leaning towards rejection", "This is inadequate",
        "Not enthusiastic", "I disagree with logic", "This is insufficient",
        "Fairly unconvinced", "Don't think right", "Somewhat negative",
        "Significant concerns", "Not really supportive", "Unpersuasive",
        "Mostly disagreeing", "Don't find merit", "Somewhat contrary",
        "Problematic overall", "Not aligned", "I disagree with conclusion",
        "This is faulty", "Fairly negative", "Don't support",
        "Somewhat opposed", "This is unreasonable", "Not convinced",
        "Fundamental doubts", "Leaning away", "Not credible",
        "Somewhat dismissive", "Don't agree with most", "This is untenable",
        "I don't think so", "Probably not", "Not really",
        "I doubt it", "Unlikely", "Not convinced",
        "Don't buy it", "Hard to believe", "Questionable claim",
        "I'm skeptical", "Not persuaded", "Seems wrong",
        "I'm doubtful", "Not likely", "Seems incorrect",
        "I'm suspicious", "Not probable", "Seems false",
        "I'm wary", "Not plausible", "Seems inaccurate",
        "I'm hesitant", "Not credible", "Seems flawed",
        "I'm reluctant", "Not believable", "Seems problematic",
        "I'm uncertain", "Not convincing", "Seems questionable",
        "I'm unconvinced", "Not valid", "Seems dubious",
        "I'm unimpressed", "Not sound", "Seems weak",
        "I'm unmoved", "Not right", "Seems off",
        "I'm disinclined", "Not accurate", "Seems wrong to me",
    ]
    
    # SCORE 1: Strong Disagreement (600 examples)
    strong_disagree_base = [
        "I completely disagree", "This is absolutely wrong", "I strongly oppose",
        "I couldn't disagree more", "This is fundamentally incorrect", "I completely reject",
        "This is totally false", "I'm entirely against", "This is completely misguided",
        "I firmly disagree", "Absolutely incorrect", "I strongly reject",
        "Entirely wrong", "I'm completely opposed", "Fundamentally flawed",
        "I totally disagree", "Utterly false", "I'm strongly against",
        "Completely unjustified", "I wholeheartedly disagree", "Absolutely unacceptable",
        "I'm firmly opposed", "Totally incorrect", "I completely refute",
        "Entirely false", "I strongly contest", "Absolutely baseless",
        "Completely against", "Fundamentally wrong", "I totally reject",
        "Utterly incorrect", "Strongly opposed", "Completely invalid",
        "I firmly reject", "Absolutely mistaken", "Entirely opposed",
        "Totally unfounded", "I strongly disapprove", "Completely erroneous",
        "I wholeheartedly reject", "Absolutely unjustified", "Firmly against",
        "Totally wrong", "I completely oppose", "Entirely baseless",
        "I strongly object", "Absolutely flawed", "Completely dissenting",
        "Fundamentally unsound", "Totally disagree", "Utterly wrong",
        "Strongly critical", "Completely unacceptable", "I firmly disagree",
        "Absolutely false", "Entirely against", "Totally invalid",
        "I strongly dispute", "Completely wrong-headed", "I wholeheartedly oppose",
        "Absolutely incorrect", "Firmly opposed", "Totally misguided",
        "I completely reject", "Entirely incorrect", "I strongly denounce",
        "Absolutely unreasonable", "Completely hostile", "Fundamentally mistaken",
        "I totally oppose", "Utterly baseless", "Strongly dismissive",
        "Completely false", "I firmly oppose", "Absolutely unsound",
        "Entirely opposed", "Totally erroneous", "I strongly refute",
        "Completely untenable", "I wholeheartedly disagree", "Absolutely wrong",
        "Firmly against", "Totally unfounded", "I completely disagree",
        "Entirely wrong", "I strongly reject", "Absolutely unacceptable",
        "Completely opposed", "Fundamentally false", "I totally reject",
        "Utterly flawed", "Strongly against", "Completely incorrect",
        "I firmly reject", "Absolutely invalid", "Entirely against",
        "Totally wrong", "I strongly oppose", "No way",
        "Definitely not", "Absolutely not", "Not at all",
        "Never", "No chance", "Impossible",
        "Out of the question", "Not happening", "Forget it",
        "No sir", "Negative", "Nope",
        "Nah", "Not a chance", "No way Jose",
        "Not in a million years", "Over my dead body", "Hell no",
        "Absolutely refuse", "I reject entirely", "Complete nonsense",
        "Total garbage", "Pure rubbish", "Utter trash",
        "Completely false", "Dead wrong", "Way off",
        "Not even close", "Couldn't be more wrong", "Totally off base",
        "Completely mistaken", "Thoroughly wrong", "Profoundly incorrect",
        "Deeply flawed", "Seriously wrong", "Badly mistaken",
        "Grossly inaccurate", "Wildly incorrect", "Patently false",
        "Manifestly wrong", "Plainly incorrect", "Obviously false",
        "Clearly wrong", "Evidently incorrect", "Undeniably false",
        "Indisputably wrong", "Unquestionably incorrect", "Categorically false",
    ]
    
    # Expand each list to 600 by repeating and adding variations
    def expand_to_600(base_list):
        expanded = []
        variations = [
            "", " really", " truly", " honestly", " definitely", " certainly",
            " absolutely", " completely", " totally", " entirely", " fully"
        ]
        
        while len(expanded) < 600:
            for item in base_list:
                if len(expanded) >= 600:
                    break
                # Add original
                expanded.append(item)
                # Add variation if not at limit
                if len(expanded) < 600:
                    variation = random.choice(variations)
                    if variation and variation not in item.lower():
                        expanded.append(item + variation)
        
        return expanded[:600]
    
    # Expand all lists
    strong_agree = expand_to_600(strong_agree_base)
    agree = expand_to_600(agree_base)
    neutral = expand_to_600(neutral_base)
    disagree = expand_to_600(disagree_base)
    strong_disagree = expand_to_600(strong_disagree_base)
    
    # Add to data
    data.extend([(text, 5) for text in strong_agree])
    data.extend([(text, 4) for text in agree])
    data.extend([(text, 3) for text in neutral])
    data.extend([(text, 2) for text in disagree])
    data.extend([(text, 1) for text in strong_disagree])
    
    # Shuffle
    random.shuffle(data)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['text', 'label'])
    
    return df

if __name__ == "__main__":
    print("Generating LARGE Likert dataset (3000 examples)...")
    df = generate_large_dataset()
    
    # Save
    df.to_csv('likert_training_data_3000.csv', index=False)
    print(f"✓ Dataset saved: {len(df)} examples")
    
    # Statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print("\nLabel distribution:")
    label_dist = df['label'].value_counts().sort_index()
    print(label_dist)
    
    print("\n=== Sample Examples ===")
    for label in [1, 2, 3, 4, 5]:
        print(f"\n--- Score {label} (Total: {label_dist[label]}) ---")
        samples = df[df['label'] == label].sample(5)
        for text in samples['text'].values:
            print(f"  • {text}")
    
    print("\n✓ Dataset generation complete!")
    print("File created: likert_training_data_3000.csv")
    print("\nNext step: Update fine_tune_likert_model.py line 107:")
    print("  Change to: load_and_prepare_data('likert_training_data_3000.csv')")