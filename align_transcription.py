import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm


SYS_PROMPT = """You are a text alignment assistant.

## Task:
- You are given a Chinese text as reference, and a english json file. You need to pick out the exact Chinese translation from the reference, to match the English text in the json file, and compose a new json output.
- DO NOT translate by yourself.
- DO NOT invent text by yourself.
- DO NOT modify any Chinese characters or punctuation.
- Copy the Chinese text EXACTLY as it appears, even if it contains line breaks.

## Output JSON only:
{
  "chunk_file": "<chunk_file>",
  "en": "<original English text>",
  "zh": "<exact Chinese text>"
}

## Examples:
### Example 1
<<<CHINESE_TEXT_BEGIN>>>
哈利波特
与魔法⽯

谨以此书献给
杰⻄卡，她喜欢这故事
安妮，她也喜欢这故事
戴，她是故事的第⼀位听众

第1章
⼤难不死的男孩
THE BOY WHO LIVED
家住⼥贞路四号的德思礼夫妇总是得意地说他们是⾮常规矩的⼈家。拜托，拜托了。他们从来跟神秘古怪的事不沾边，因为他们根本不相信那些邪⻔歪道。
弗农·德思礼先⽣在⼀家名叫格朗宁的公司做主管，公司⽣产钻机。他⾼⼤魁梧，胖得⼏乎连脖⼦都没了，却蓄着⼀脸⼤胡⼦。德思礼太太是个瘦削的⾦发⼥⼈。她的脖⼦⼏乎⽐正常⼈⻓⼀倍。这样每当她花许多时间隔着篱墙引颈⽽望、窥探左邻右舍时，她的⻓脖⼦可就派上了⼤⽤场。德思礼夫妇有⼀个⼩⼉⼦，名叫达⼒。在他们看来，⼈世间没有⽐达⼒更好的孩⼦了。
德思礼⼀家什么都不缺，但他们拥有⼀个秘密，他们最害怕的就是这秘密会被⼈发现。他们想，⼀旦有⼈发现波特⼀家的事，他们会承受不住的。波特太太是德思礼太太的妹妹，不过她们已经有好⼏年不⻅⾯了。实际上，德思礼太太佯装⾃⼰根本没有这么个妹妹，因为她妹妹和她那⼀⽆是处的妹夫与德思礼⼀家的为⼈处世完全不⼀样。⼀想到邻居们会说波特夫妇来到了，德思礼夫妇会吓得胆战⼼惊。他们知道波特也有个⼉⼦，只是他们从来没有⻅过。这孩⼦也是他们不与波特夫妇来往的⼀个很好的借⼝，他们不愿让达⼒跟这种孩⼦厮混。
我们的故事开始于⼀个晦暗、阴沉的星期⼆，德思礼夫妇⼀早醒来，窗外浓云低垂的天空并没有丝毫迹象预示这地⽅即将发⽣神秘古怪的事情。德思礼先⽣哼着⼩曲，挑出⼀条最不喜欢的领带戴着上班，德思礼太太⾼⾼兴兴，⼀直絮絮叨叨，把唧哇乱叫的达⼒塞到了⼉童椅⾥。
他们谁也没留意⼀只⻩褐⾊的猫头鹰扑扇着翅膀从窗前⻜过。
⼋点半，德思礼先⽣拿起公⽂包，在德思礼太太⾯颊上亲了⼀下，正要亲达⼒，跟这个⼩家伙道别，可是没有亲成，⼩家伙正在发脾⽓，把⻨⽚往墙上摔。“臭⼩⼦。”德思礼先⽣嘟哝了⼀句，咯咯笑着⾛出家⻔，坐进汽⻋，倒出四号⻋道。
在街⻆上，他看到了第⼀个异常的信号——⼀只猫在看地图。⼀开始，德思礼先⽣还没弄明⽩他看到了什么，于是⼜回过头去。只⻅⼀只花斑猫站在⼥贞路的路⼝，但是没有看⻅地图。他到底在想些什么？很可能是光线使他产⽣了错觉吧。德思礼先⽣眨了眨眼，盯着猫着，猫也瞪着他。当德思礼先⽣拐过街⻆继续上路的时候，他从后视镜⾥看看那只猫。猫这时正在读⼥贞路的标牌，不，是在看标牌；猫是不会读地图或是读标牌的。德思礼先⽣定了定神，把猫从脑海⾥赶⾛。他开⻋进城，⼀路上想的是希望今天他能得到⼀⼤批钻机的定单。
但快进城时，另⼀件事⼜把钻机的事从他脑海⾥赶⾛了。当他的⻋汇⼊清晨拥堵的⻋流时，他突然看⻅路边有⼀群穿着奇装异服的⼈。他们都披着⽃篷。德思礼先⽣最看不惯别⼈穿得怪模怪样，瞧年轻⼈的那身打扮！他猜想这⼤概⼜是⼀种⽆聊的新时尚吧。他⽤⼿指敲击着⽅向盘，⽬光落到了离他最近的⼀⼤群怪物身上。他们正兴致勃勃，交头接⽿。德思礼先⽣很⽣⽓，因为他发现他们中间有⼀对根本不年轻了，那个男的显得⽐他年龄还⼤，竟然还披着⼀件翡翠绿的⽃篷！真不知羞耻！接着，德思礼先⽣突然想到这些⼈⼤概是为什么事募捐吧，不错，就是这么回事。⻋流移动了，⼏分钟后德思礼先⽣来到了格朗宁公司的停⻋场，他的思绪⼜回到了钻机上。
德思礼先⽣在他⼗楼的办公室⾥，总是习惯背窗⽽坐。如果不是这样，他可能会发现这⼀天早上他更难把思想集中到钻机的事情上了。他没有看⻅成群的猫头鹰在光天化⽇之下从天上⻜过，可街上的⼈群都看到了；他们⽬瞪⼝呆，指指点点，盯着猫头鹰⼀只接⼀只从头顶上掠过。他们⼤多甚⾄夜⾥都从未⻅过猫头鹰。德思礼先⽣这天早上很正常，没有受到猫头鹰的⼲扰。他先后对五个⼈⼤喊⼤叫了⼀遍，⼜打了⼏个重要的电话，喊的声⾳更响。他的情绪很好，到吃午饭的时候，他想舒展⼀下筋⻣，到⻢路对⻆的⾯包房去买⼀只⼩甜圆⾯包。
若不是他在⾯包房附近⼜碰到那群披⽃篷的⼈，他早就把他们忘了。他经过他们身边时，狠狠地瞪了他们⼀眼。他说不清这是为什么，只是觉得这些⼈让他⼼⾥别扭。这些⼈正嘁嘁喳喳，讲得起劲，但他连⼀只募捐箱也没有看⻅。当他拎着装在袋⾥的⼀只⼤油饼往回⾛，经过他们身边时，他们的话断断续续飘⼊他的⽿⿎：
“波特夫妇，不错，我正是听说……”
“……没错，他们的⼉⼦，哈利……”
<<<CHINESE_TEXT_END>>>

<<<ENGLISH_JSON_BEGIN>>>
{
  "chunk_file": "chunk_1.wav",
  "text": " Harry Potter and the Philosopher's Stone by J. K. Rowling Chapter 1 The Boy Who Lived",
  "language": "en"
}
<<<ENGLISH_JSON_END>>>


<<<OUTPUT_JSON_BEGIN>>>
{
  "chunk_file": "chunk_1.wav",
  "en": " Harry Potter and the Philosopher's Stone by J. K. Rowling Chapter 1 The Boy Who Lived",
  "zh": "哈利波特\n与魔法⽯\n第1章\n⼤难不死的男孩"
}
<<<OUTPUT_JSON_END>>>"""


def extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in output.")
    return json.loads(text[start : end + 1])


def remove_linebreaks_between_chinese(text: str):
    # Remove \n only when between Chinese chars
    pattern = r"([\u4e00-\u9fff])\s*\n\s*([\u4e00-\u9fff])"
    while re.search(pattern, text):
        text = re.sub(pattern, r"\1\2", text)
    return text


def align_chunk(reference_file: str, json_file: str, pipe, tokenizer):
    # Load inputs
    ref_path = Path(reference_file)
    title_path = ref_path.parent / "title.txt"
    chinese_text = ref_path.read_text(encoding="utf-8")
    if "_1.json" in json_file:
        title_text = Path(title_path).read_text(encoding="utf-8")
        chinese_text = title_text + "\n\n" + chinese_text
    metadata = json.loads(Path(json_file).read_text(encoding="utf-8"))
    json_text = json.dumps(metadata, ensure_ascii=False, indent=2)

    # Build prompt
    system_prompt = SYS_PROMPT

    user_prompt = f"""Here is the Chinese reference text:

<<<CHINESE_TEXT_BEGIN>>>
{chinese_text}
<<<CHINESE_TEXT_END>>>


Here is the English JSON:
<<<ENGLISH_JSON_BEGIN>>>
{json_text}
<<<ENGLISH_JSON_END>>>

Now, output the aligned json.
Note: 
Extract ONLY the matching Chinese text.
Return ONLY the final JSON object."""

    messages = [
        {"role": "system", "content": system_prompt},
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
        max_new_tokens=2048,
        return_full_text=False,
    )[0]["generated_text"]

    # Extract JSON
    result = extract_json(output)

    # Overwrite metadata
    result["chunk_file"] = metadata["chunk_file"]
    result["en"] = metadata["text"]

    # Clean Chinese formatting safely
    result["zh"] = remove_linebreaks_between_chinese(result["zh"])

    # Save
    output_file = json_file.replace(".json", "_aligned.json")
    Path(output_file).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved to", output_file)
    return result


def align_chunks(
    reference_file: str, chunks_folder: str, model_name: str = "Qwen/Qwen3-8B"
):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    chunk_files = sorted(Path(chunks_folder).glob("*.json"))
    for chunk_file in tqdm(chunk_files):
        try:
            print("Processing", chunk_file)
            align_chunk(
                reference_file=reference_file,
                json_file=str(chunk_file),
                pipe=pipe,
                tokenizer=tokenizer,
            )
        except Exception as e:
            print(f"Error processing {chunk_file}: {e}")
            output_file = str(chunk_file).replace(".json", "_error.json")
            Path(output_file).write_text(
                json.dumps({}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue


if __name__ == "__main__":
    align_chunks(
        reference_file="/home/bo/workspace/whisper/tasks/HP1/text_zh/1/ch1.txt",
        chunks_folder="/home/bo/workspace/whisper/tasks/HP1/audio_en/ch1_chunks",
        model_name="Qwen/Qwen3-8B",
    )
