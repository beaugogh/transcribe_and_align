import os
import json
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json5
from json_repair import repair_json


def read_lines(filepath, encoding="utf-8"):
    """
    Reads all non-empty lines from a text file and returns them as a clean list.
    Empty or whitespace-only lines are skipped.
    """
    lines = []
    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            clean = line.strip()
            if clean:  # keep only non-empty lines
                lines.append(clean)
    return lines


def get_reference(i):
    offset = 50
    x = len(lines) * i / 77
    print(x)
    indx = int(x)
    s, e = max(0, indx - offset), indx + offset
    print(s, e)
    return "\n".join(lines[s : e + 1])


def find_text():
    for i, l in enumerate(lines):
        if "receiver back down and stroked" in l:
            print(i)
            print(l)


def read_json(filepath, encoding="utf-8"):
    with open(filepath, "r", encoding=encoding) as f:
        data = json.load(f)
    return data


def align_chunk(chunk_indx, source_text, tokenizer, pipe):
    reference = get_reference(chunk_indx)
    sys_prompt = """You are an expert at textual alignment. 
You are given a piece of text as source, and a reference paragraph, your task is to find and select the target text from the reference paragraph that matches the source.
Note that the source and the target may vary slightly, but overall almost identical. You MUST NOT invent new texts, you just need to find the matching target text from the reference paragraph, and faithfully copy & paste it as the target. You MUST NOT over-select or under-select target text from the reference paragraph. 
You MUST NOT add redundant words to your target output, just the matching text your found in the reference, nothing more, nothing less. 
A simple rule of thumb is that the beginning and the end of the target text must closely match the beginning and the end of the source text respectively.
Output your thinking process (no more than 100 tokens) and the target text in correct json format.

## example
Input:
{
    "source": "\"'Mr. and Mrs. Dursley, of No. 4 Privet Drive, \"'were proud to say that they were perfectly normal, thank you very much. ",
    "reference": "CHAPTER ONE\nTHE BOY WHO LIVED\nMr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense.\nMr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache. Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere.\nThe Dursleys had everything they wanted, but they also had a secret, and their greatest fear was that somebody would discover it. They didn’t think they could bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley’s sister, but they hadn’t met for several years; in fact, Mrs. Dursley pretended she didn’t have a sister, because her sister and her good-for-nothing husband were as unDursleyish as it was possible to be. The Dursleys shuddered to think what the neighbors would say if the Potters arrived in the street. The Dursleys knew that the Potters had a small son, too, but they had never even seen him. This boy was another good reason for keeping the Potters away; they didn’t want Dudley mixing with a child like that."
}
Output:
{
    "think": "The punctuation artificats like \" is of no important, I should find text that starts with Mr and Mrs Dursley and ends with thank you very much. Anything outside this boundary, I just ignore.",
    "target": "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. "
}

## example
Input:
{
    "source": "He put the receiver back down and stroked his moustache, thinking.",
    "reference": "Mr. Dursley stopped dead. Fear flooded him. He looked back at the whisperers as if he wanted to say something to them, but thought better of it. He dashed back across the road, hurried up to his office, snapped at his secretary not to disturb him, seized his telephone, and had almost finished dialing his home number when he changed his mind. He put the receiver back down and stroked his mustache, thinking…no, he was being stupid. Potter wasn’t such an unusual name. He was sure there were lots of people called Potter who had a son called Harry. Come to think of it, he wasn’t even sure his nephew was called Harry. He’d never even seen the boy. It might have been Harvey. Or Harold. There was no point in worrying Mrs. Dursley; she always got so upset at any mention of her sister. He didn’t blame her — if he’d had a sister like that…but all the same, those people in cloaks.… He found it a lot harder to concentrate on drills that afternoon and when he left the building at five o’clock, he was still so worried that he walked straight into someone just outside the door."
}
Output:
{
    "think": "The source text is clean, no weird punctuation or other artifacts. I should seek the part of the reference that starts with He put..., and ends with thinking". I find this piece of text in the reference! Note that the in the target text, mustache is used instead of moustache in the source text, which is ok. I must stick to the wording in the reference.",
    "target": "He put the receiver back down and stroked his mustache, thinking."
}
"""
    user_prompt = f"""Here is the source and reference texts, give your thinking process and find the target:
Input:
{{
    "source": "{source_text}",
    "reference": "{reference}"
}}
Output:
"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # print(user_prompt)
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Run inference
    output = pipe(
        prompt,
        temperature=0.1,
        top_p=0.8,
        max_new_tokens=2048,
        return_full_text=False,
    )[0]["generated_text"]
    output = repair_json(output)
    output = json5.loads(output)
    return output


def verify_alignment(res_json, tokenizer, pipe):
    sys_prompt = """You are an expert at textual alignment. 
You are given a piece of json text, compare the source and target texts, and judge if the two are aligned. And output a boolean judgement, along with your thinking process (no more than 100 tokens) in a json format.

It is ok for the two to have different spelling/punctuation here and there.
But it is NOT ok if one text contains a whole chunk of words more than the other. Below are two examples.

## example 1
Input:
{
    "chunk_file": "chunk_10.wav", 
    "source": " There was a tabby cat standing on the corner of Privet Drive, but there wasn't a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr Dursley blinked and stared at the cat. It stared back. As Mr Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive. No, looking at the sign. Cats couldn't read maps or signs. Mr Dursley gave himself a little shake and put the cat out of his mind.", 
    "target": "There was a tabby cat standing on the corner of Privet Drive, but there wasn’t a map in sight. What could he have been thinking of? It must have been a trick of the light. Mr. Dursley blinked and stared at the cat. It stared back. As Mr. Dursley drove around the corner and up the road, he watched the cat in his mirror. It was now reading the sign that said Privet Drive — no, looking at the sign; cats couldn’t read maps or signs. Mr. Dursley gave himself a little shake and put the cat out of his mind."
}
Output:
{
    "think": "the source has one extra leading space than the target, which is ok, the rest is the same as the target.",
    "decision": "True"
}

## example 2
Input:
{
    "chunk_file": "chunk_10.wav", 
    "source": " There was a tabby cat standing on the corner of Privet Drive, but there wasn't a map in sight. ", 
    "target": "There was a tabby cat standing on the corner of Privet Drive,"
}
Output:
{
    "think": "the target has this whole part missing compared to the source, i.e. ' but there wasn't a map in sight. ', so this is an incorrect matchup.",
    "decision": "False"
}

## example 3
Input:
{
    "chunk_file": "chunk_1.wav",
    "source": " Harry Potter and the Philosopher's Stone by J. K. Rowling Chapter 1 The Boy Who Lived",
    "target": "CHAPTER ONE\nTHE BOY WHO LIVED"
}
Output:
{
    "think": "the source and the target are vastly different, the 'Harry Potter and the Philosopher's Stone by J. K. Rowling' part is entirely missing in the target",
    "decision": "False"
}

## example 4:
Input:
{
  "chunk_file": "chunk_8.wav",
  "source": " At half-past eight, Mr Dursley picked up his briefcase, pecked Mrs Dursley on the cheek, and tried to kiss Dudley goodbye, but missed, because Dudley was now having a tantrum and throwing his cereal at the walls. \"'Oh, little tyke!' chortled Mr Dursley as he left the house. He got into his car and backed out of No. 4's drive.",
  "target": "At half past eight, Mr. Dursley picked up his briefcase, pecked Mrs. Dursley on the cheek, and tried to kiss Dudley good-bye but missed, because Dudley was now having a tantrum and throwing his cereal at the walls."
}
Output:
{
    "think": "it is almost correct, except for the last part: Oh, little tyke!' chortled Mr Dursley, which the target text is missing",
    "decision": "False"
}

## example 5:
Input:
{
  "chunk_file": "chunk_50.wav",
  "source": " \"'I suppose he really has gone, Dumbledore?' \"'It certainly seems so,' said Dumbledore. \"'We have much to be thankful for. Would you care for a sherbet lemon?' \"'A what?' \"'A sherbet lemon. They're a kind of muggle sweet I'm rather fond of.' \"'No, thank you,' said Professor McGonagall, coldly, as though she didn't think this was the moment for sherbet lemons. \"'As I say, even if you-know-who has gone, my dear Professor, surely a sensible person like yourself can call him by his name? All this you-know-who nonsense! For eleven years I've been trying to persuade people to call him by his proper name. Voldemort!' Professor McGonagall flinched, but Dumbledore, who was unsticking two sherbet lemons, seemed not to notice. \"'It all gets so confusing. If we keep saying you-know-who, I've never seen any reason to be frightened of saying Voldemort's name.' \"'I know you haven't,' said Professor McGonagall, sounding half-exasperated, half-admiring. \"'But you're different. Everyone knows you're the only one you know—' \"'Oh, all right. Voldemort was frightened of.' \"'You flatter me,' said Dumbledore, calmly. \"'Voldemort had powers I will never have. Only because you—' \"'You're too, well, noble to use them.' \"'Oh, it's lucky it's dark. I haven't blushed so much since Madam Pumphrey told me she liked my new earmuffs.' Professor McGonagall shot a sharp look at Dumbledore and said, \"'The owls are nothing to the rumours that are flying around. You know what everyone's saying about why he's disappeared, about what finally stopped him?'",
  "target": "Only because you’re too — well — noble to use them. It’s lucky it’s dark. I haven’t blushed so much since Madam Pomfrey told me she liked my new earmuffs. Professor McGonagall shot a sharp look at Dumbledore and said “The owls are nothing next to the rumors that are flying around. You know what they’re saying? About why he’s disappeared? About what finally stopped him?”"
}
Output:
{
    "think": "this is obviously wrong, even the first words are different: I vs Only",
    "decision": "False"
}


"""
    user_prompt = f"""Here is input json data, give your thinking process and decision in strictly json format in the output:
Input:
{res_json}
Output:

"""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Run inference
    output = pipe(
        prompt,
        temperature=0.1,
        top_p=0.8,
        max_new_tokens=1024,
        return_full_text=False,
    )[0]["generated_text"]

    output = json5.loads(output)
    return output


if __name__ == "__main__":

    lines = read_lines(
        "/home/bo/workspace/transcribe_and_align/data/HP1/text_en/1/ch1.txt"
    )
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    folder = "/home/bo/workspace/transcribe_and_align/data/HP1/audio_en/ch1_chunks"
    files = [
        f for f in sorted(os.listdir(folder)) if ".json" in f and "aligned" not in f
    ]
    for f in files:
        p = os.path.join(folder, f)
        p_aligned = p.replace(".json", "_aligned.json")
        if not os.path.exists(p_aligned):
            print(f)
            data = read_json(p)
            source = data["text"]
            chunk_file = data["chunk_file"]
            indx = int(f.replace(".json", "").replace("chunk_", ""))
            a_output = align_chunk(indx, source, tokenizer, pipe)
            pprint(a_output)
            res = {
                "chunk_file": chunk_file,
                "source": source,
                "target": a_output["target"],
            }
            pprint(res)
            res_json = json.dumps(res, ensure_ascii=False, indent=2)
            b_output = verify_alignment(res_json, tokenizer, pipe)
            pprint(b_output)
            if b_output["decision"].lower() == "true":
                output_file = os.path.join(
                    folder, chunk_file.replace(".wav", "_aligned.json")
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(res, f, ensure_ascii=False, indent=2)

            print(
                "======================================================================"
            )
