import pickle
import json
import argparse

def load_pkl(file_path):
    with open(file_path, 'rb') as fpkl:
        data = pickle.load(fpkl)
        print(data)

def convert_dict_to_json(file_path):
    with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w') as fjson:
        data = pickle.load(fpkl)
        json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Read PKL Files')
    parser.add_argument('file', help='file path')
    args = parser.parse_args()
    pkl_file = args.file
    try:
        convert_dict_to_json(pkl_file)
    except:
        load_pkl(pkl_file)

if __name__ == '__main__':
    main()