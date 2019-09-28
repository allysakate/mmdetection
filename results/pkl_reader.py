import pickle
import json


def convert_dict_to_json(file_path):
    with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w') as fjson:
        data = pickle.load(fpkl)
        json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4)

def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('file', help='result file path')
    args = parser.parse_args()
    pkl_file = args.file
    convert_dict_to_json(pkl_file)

if __name__ == '__main__':
    main()