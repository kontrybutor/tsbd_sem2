
import apache_log_parser
from pprint import pprint

INPUT_FILE_NAME = "log.txt"
OUTPUT_FILE_NAME = "parsed_log.txt"
NUMBER_OF_LINES = 50000


class Parser(object):

    filtered_lines = []
    list_of_extracted_lines = []

    def preprocess_log(self):
        """ Helper function, takes input file and extract first 50000 lines into new file. """
        with open(INPUT_FILE_NAME, 'r') as input_file:
            with open(OUTPUT_FILE_NAME, 'w') as output_file:
                for _ in range(NUMBER_OF_LINES):
                    output_file.write(input_file.readline())

    def filter_log(self):
        """TBD..."""

        with open(OUTPUT_FILE_NAME) as processed_file:
            lines = [next(processed_file) for _ in range(NUMBER_OF_LINES)]
            print("Before filtering is", len(lines), "lines.")
            for line in lines:
                # TODO: refactor this sometime...
                if "GET" in line and "200" in line and ".jpg" not in line and ".gif" not in line and ".bmp" not in line:
                    self.filtered_lines.append(line)

        print("After filtering is", len(self.filtered_lines), "lines.")

    def extract_values_into_dict(self):
        """TBD..."""

        line_parser = apache_log_parser.make_parser("%h %t \"%r\" %>s %b \"%{User-Agent}i\" %l %u")
        for line in self.filtered_lines:
            extracted_data_from_line = line_parser(line)
            self.list_of_extracted_lines.append(extracted_data_from_line)
        # pprint(self.list_of_extracted_lines)

    def count_occurences(self):
        pass


def main():
    parser = Parser()

    parser.filter_log()
    parser.extract_values_into_dict()


if __name__ == "__main__":
    main()
