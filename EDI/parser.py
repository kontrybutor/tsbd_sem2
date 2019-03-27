import apache_log_parser
from pprint import pprint
from collections import Counter

INPUT_FILE_NAME = "log.txt"
OUTPUT_FILE_NAME = "parsed_log.txt"
NUMBER_OF_LINES = 50


def preprocess_log():
    """ Helper function, takes input file and extract first 50000 lines into new file. """
    with open(INPUT_FILE_NAME, 'r') as input_file:
        with open(OUTPUT_FILE_NAME, 'w') as output_file:
            for _ in range(NUMBER_OF_LINES):
                output_file.write(input_file.readline())


class Parser(object):
    filtered_lines = []
    list_of_extracted_lines = []

    def filter_log(self):
        """TBD..."""
        media_exts = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.gif', '.GIF', '.bmp', '.BMP', '.png', '.PNG', '.mpg', '.MPG']
        with open(OUTPUT_FILE_NAME) as processed_file:
            lines = [next(processed_file) for _ in range(NUMBER_OF_LINES)]
            print("Before filtering are", len(lines), "lines.")
            for line in lines:
                if "GET" in line and "200" in line:
                    if any(word in line for word in media_exts):
                        continue
                    else:
                        self.filtered_lines.append(line)

        print("After filtering there are", len(self.filtered_lines), "lines.")

    def extract_values_into_dict(self):
        """TBD..."""

        keys = ['remote_host', 'request_url', 'time_received_datetimeobj', 'request_method', 'request_first_line', ]
        line_parser = apache_log_parser.make_parser("%h %t \"%r\" %s %b")
        for line in self.filtered_lines:
            extracted_data_from_line = line_parser(line)
            filtered_extracted_values = dict(
                (k, extracted_data_from_line[k]) for k in keys if k in extracted_data_from_line)
            self.list_of_extracted_lines.append(filtered_extracted_values)

    def count_occurrences(self, key):

        total_count = 0
        occurrences = Counter(k[key] for k in self.list_of_extracted_lines if k.get(key))
        for occurrence, count in occurrences.most_common():
            # print(occurrence, count)
            total_count += count

        print("Całkowita liczba wynosi:", total_count)

    def extract_distinct_users(self):

        pprint(self.list_of_extracted_lines)
        distinct_user_list = []
        for elem in self.list_of_extracted_lines:
            if elem['remote_host'] not in distinct_user_list:
                distinct_user_list.append(elem['remote_host'])
        pprint(distinct_user_list)


def main():
    # preprocess_log()

    parser = Parser()

    parser.filter_log()
    parser.extract_values_into_dict()
    parser.count_occurrences('remote_host')
    parser.count_occurrences('request_url')
    parser.extract_distinct_users()


if __name__ == "__main__":
    main()
