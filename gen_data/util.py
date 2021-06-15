FPT_VOICE = {
    'leminh': {
        'gender': 'nam',
        'region': 'bac',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    },
    'banmai': {
        'gender': 'nu',
        'region': 'bac',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    },
    'thuminh': {
        'gender': 'nu',
        'region': 'bac',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    },
    'giahuy': {
        'gender': 'nam',
        'region': 'trung',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    },
    'ngoclam': {
        'gender': 'nu',
        'region': 'trung',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    },
    'myan': {
        'gender': 'nu',
        'region': 'trung',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    },
    'lannhi': {
        'gender': 'nu',
        'region': 'nam',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    },
    'linhsan': {
        'gender': 'nu',
        'region': 'nam',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    },
    'minhquang': {
        'gender': 'nam',
        'region': 'nam',
        'speed': [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
    }
}


def get_voices(args):
    if args.api == 'fpt':
        for name in FPT_VOICE:
            gender = FPT_VOICE[name]['gender']
            region = FPT_VOICE[name]['region']
            for speed in FPT_VOICE[name]['speed']:
                yield name, gender, region, speed