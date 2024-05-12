import os
import sys

from tqdm import tqdm

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python get_list_file.py $DATASET_ROOT")
        sys.exit(0)
    else:
        dataset_root = sys.argv[1]
    list_file = os.path.join(dataset_root, 'dataset_index.txt')
    with open(list_file, 'w') as f:
        scenarios = os.listdir(dataset_root)
        for scenario in tqdm(scenarios):
            route = os.path.join(dataset_root, scenario)
            if os.path.isdir(route):
                frames = len(os.listdir(os.path.join(route, 'measurements')))
                if frames < 32:
                    print("Route %s only havs %d frames (<32). We have omitted it!" % (route, frames))
                else:
                    f.write(route + ' ' + str(frames) + '\n')
