

INPUT_FILE_NAME = "log.txt"
OUTPUT_FILE_NAME = "parsed_log.txt"
NUMBER_OF_LINES = 50000
head = []


class Parser(object):

    def preprocess_log(self):
        """ Helper function, takes input file and extract first 50000 lines into new file. """
        with open(INPUT_FILE_NAME, 'r') as input_file:
            with open(OUTPUT_FILE_NAME, 'w') as output_file:
                for _ in range(NUMBER_OF_LINES):
                    output_file.write(input_file.readline())

    def filter_log(self):
        """TBD..."""
        filtered_logs = []
        pos = ["GET", "200", ]
        neg = [".jpg" ".gif", ".bmp", ".xmb"]
        with open(OUTPUT_FILE_NAME) as processed_file:
            lines = [next(processed_file) for _ in range(NUMBER_OF_LINES)]
            print("Before filtering is", len(lines), "lines.")
            for line in lines:
                # if any(x in line for x in pos) and any(x not in line for x in neg):
                # TODO: refactor this in more pythonic way...
                if "GET" in line and "200" in line:
                    filtered_logs.append(line)

        print("After filtering is", len(filtered_logs), "lines.")


def main():
    parser = Parser()

    parser.filter_log()


if __name__ == "__main__":
    main()
