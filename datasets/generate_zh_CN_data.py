import os
import random
import soundfile as sf
from datasets import load_dataset
data_loaded = load_dataset('fixie-ai/covost2', 'zh-CN_en', 
                               split=['train', 'validation', 'test'])
os.makedirs('covost2_zh-CN',exist_ok=True)
for ds in data_loaded:
    for sample in ds:
        audio_meta=sample['audio']
        path = audio_meta['path']
        sample_rate = audio_meta['sampling_rate']
        audio_data= audio_meta['array']
        length = len(audio_data)
        if length <= 1:
            continue
        max_truncate_length = length - 1
        truncate_length = random.randint(1, max_truncate_length)
        truncated_audio = audio_data[:truncate_length]
        path=path.replace('mp3','wav')
        sf.write('covost2_zh-CN/positive'+path, audio_data, sample_rate, subtype='FLOAT')
        sf.write('covost2_zh-CN/negative'+path, truncated_audio, sample_rate, subtype='FLOAT')
