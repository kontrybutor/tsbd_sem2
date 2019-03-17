import apache_log_parser
from pprint import pprint
from collections import Counter

INPUT_FILE_NAME = "log.txt"
OUTPUT_FILE_NAME = "parsed_log.txt"
NUMBER_OF_LINES = 500


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

        with open(OUTPUT_FILE_NAME) as processed_file:
            lines = [next(processed_file) for _ in range(NUMBER_OF_LINES)]
            print("Before filtering are", len(lines), "lines.")
            for line in lines:
                # TODO: refactor this sometime...
                if "GET" in line and "200" in line and ".jpg" not in line and ".gif" not in line and ".bmp" not in line:
                    self.filtered_lines.append(line)

        print("After filtering are", len(self.filtered_lines), "lines.")

    def extract_values_into_dict(self):
        """TBD..."""

        keys = ['remote_host', 'request_url', 'time_received_datetimeobj', 'request_method', 'request_first_line', ]
        line_parser = apache_log_parser.make_parser("%h %t \"%r\" %s %b")
        for line in self.filtered_lines:
            extracted_data_from_line = line_parser(line)
            filtered_extracted_values = dict(
                (k, extracted_data_from_line[k]) for k in keys if k in extracted_data_from_line)
            self.list_of_extracted_lines.append(filtered_extracted_values)
        # pprint(self.list_of_extracted_lines)

    def count_occurences(self, key):

        print("Całkowita liczba", key, "wynosi", len(self.list_of_extracted_lines))
        print("Całkowita liczba", key, "wynosi", len(self.list_of_extracted_lines))
        print("Dla klucza:", key)

        total_count = 0
        occurrences = Counter(k[key] for k in self.list_of_extracted_lines if k.get(key))
        for occurrence, count in occurrences.most_common():
            print(occurrence, count)
            total_count += count

        print("Całkowita liczba wynosi:", total_count)


def main():

    # preprocess_log()

    parser = Parser()

    parser.filter_log()
    parser.extract_values_into_dict()
    parser.count_occurences('remote_host')
    parser.count_occurences('request_url')


if __name__ == "__main__":
    main()
