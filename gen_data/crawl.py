import argparse
import requests
import json
import time
import tqdm
import os
from scipy.io.wavfile import write
import numpy as np

import util

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep', type=int, default=1)
    parser.add_argument('--token_name', type=str, default='token')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--api', type=str, default='fpt')
    parser.add_argument('--output_dir', type=str, default='data')
    args = parser.parse_args()
    args.url = 'https://api.fpt.ai/hmi/tts/v5'
    return args



def generate_script(args):
    script_txt = os.path.join(args.data_dir, 'script.txt')
    script_json = os.path.join(args.data_dir, 'script.json')

    scripts = open(script_txt).read().strip().split('\n')
    i = 0
    with open(script_json, 'w', encoding='utf-8') as fp:
        for s in tqdm.tqdm(scripts):
            for name, gender, region, speed in util.get_voices(args):
                meta = {
                    'name': name,
                    'gender': gender,
                    'region': region,
                    'speed': speed,
                    'filename': f'{i}_{name}_{gender}_{region}',
                    'text': s.split('|')[1],
                    'folder': s.split('|')[0],
                }
                json.dump(meta, fp, ensure_ascii=False)
                fp.write('\n')
                i += 1


def generate_token(args):
    token_csv = os.path.join(args.data_dir, f'{args.token_name}.csv')
    token_json = os.path.join(args.data_dir, f'{args.token_name}.json')
    tokens = [l.split(',') for l in open(token_csv).read().strip().split('\n')]
    
    if not os.path.exists(token_json):
        meta = {}
        for token in tokens:
            meta[token] = {
                'username': 'ceo@vietott.org',
                'remain': 5000,
            }

        with open(token_json, 'w', encoding='utf-8') as fp:
            json.dump(meta, fp, indent=4)
    else:
        meta = json.loads(open(token_json).read())
        # for token in tokens:
        #     meta[token] = {
        #         'username': 'minhairtran@gmail.com',
        #         'remain': 5000,
        #     }
        with open(token_json, 'w', encoding='utf-8') as fp:
            json.dump(meta, fp, indent=4)


def load_script(args):
    script_json = os.path.join(args.data_dir, 'script.json')
    for meta in open(script_json):
        yield json.loads(meta)

def load_token(args):
    token_json = os.path.join(args.data_dir, f'{args.token_name}.json')
    return json.loads(open(token_json).read())

def update_token(meta, args):
    token_json = os.path.join(args.data_dir, f'{args.token_name}.json')
    with open(token_json, 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, indent=4)

def choose_token(num_character, tokens):
    for t in tokens:
        if tokens[t]['remain'] >= num_character:
            return t
    return None


def call(meta, tokens, args):
    name = meta['name']
    gender = meta['gender']
    region = meta['region']
    speed = meta['speed']
    text = meta['text']
    filename = meta['filename']
    num_character = len(list(text))
    dirname = os.path.join(args.output_dir, args.api, meta['folder'])


    if not os.path.exists(dirname):
        os.makedirs(dirname)

    wavname = os.path.join(dirname, filename + '.wav')
    txtname = os.path.join(dirname, filename + '.txt')

    if os.path.exists(wavname):
        return

    token = choose_token(num_character, tokens)
    if not token:
        print('[+] Done')
        return

    print(f'    text: {text}')
    print(f'    voice: {name}, region: {region}, gender: {gender}, speed: {speed}')
    print(f'    num_character: {num_character}')
    print(f'    token: {token}')
    print(f'    dirname: {dirname}')
    print(f'    wavname: {wavname}')
    print(f'    txtname: {txtname}')

    headers = {
        'api-key': token,
        'speed': str(speed),
        'voice': name
    }
    try:    
        response = requests.post(args.url, data=text.encode('utf-8'), headers=headers)
    except KeyboardInterrupt:
        return
    except:
        print(    f'[!] error in requesting {args.api} api')
        return
    else:
        print(f'    get status_code: {response.status_code}')
        if response.status_code == 200:
            try:
                print('    updating tokens')
                tokens[token]['remain'] -= num_character
                update_token(tokens, args)

                link = json.loads(response.content)['async']
                print(f'    link: {link}')
                for i in range(30):
                    time.sleep(args.sleep)
                    download = requests.get(link)
                    print(f'    download status_code: {download.status_code}')

                    if download.status_code == 200:
                        print('    saving wav file')
                        with open(wavname, 'wb') as f:
                            f.write(download.content)
                        # with open(txtname, 'w') as f:
                        #     f.write(text)
                        #     f.write('\n')
                        time.sleep(args.sleep)
                        return 

            except KeyboardInterrupt:
                print('    updating tokens')
                tokens[token]['remain'] -= num_character
                update_token(tokens, args)
                return
            except:
                print(    '[!] error in downloading wav')
                print(json.loads(response.content))
                return


def main():
    args = get_args()

    print('[+] Generating scripts')
    generate_script(args)

    print('[+] Loading scripts')
    scripts = list(load_script(args))[::-1]

    print('[+] Generating tokens')
    generate_token(args)

    print('[+] Loading tokens')
    tokens = load_token(args)

    print('[+] Processing')
    for i, meta in enumerate(scripts):
        print(f'============================ {i} ============================')
        call(meta, tokens, args)
        # exit()

if __name__ == '__main__':
    main()