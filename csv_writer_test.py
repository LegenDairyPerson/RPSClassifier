import csv
import re
import os

_NAME_RE = re.compile(
    r"^(custom-rps|custom-test-set)(?:/|\\)(rock|paper|scissors)(?:/|\\)[\w-]*\.png$"
)

file = open("custom-test-set/custom_test_labels.csv", 'w', newline='')

writer = csv.writer(file)

csv_list = []

def process_directory(directory, folder):
    for filename in os.listdir(os.path.join(directory, folder)):
        file_path = os.path.join(directory, folder, filename)
        print(file_path)
        if os.path.isfile(file_path):
            # Check if the filename matches the specified pattern
            res = _NAME_RE.match(file_path)
            if not res:
                continue

            image = file_path
            label = res.group(2).lower()

            csv_list.append((image, label))


process_directory("custom-test-set", "paper")
process_directory("custom-test-set", "rock")
process_directory("custom-test-set", "scissors")

for image, label in csv_list:
    writer.writerow([image, label])

file.close()
