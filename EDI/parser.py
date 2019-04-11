import apache_log_parser
from pprint import pprint
from collections import Counter, defaultdict
from datetime import datetime, MINYEAR, timedelta
INPUT_FILE_NAME = "log.txt"
OUTPUT_FILE_NAME = "parsed_log.txt"
NUMBER_OF_LINES = 50000


def preprocess_log():
    """ Helper function, takes input file and extract first 50000 lines into new file. """
    with open(INPUT_FILE_NAME, 'r') as input_file:
        with open(OUTPUT_FILE_NAME, 'w') as output_file:
            for _ in range(NUMBER_OF_LINES):
                output_file.write(input_file.readline())


class Parser(object):

    def __init__(self):
        self.filtered_lines = []
        self.list_of_extracted_lines = []
        self.distinct_user_list = []
        self.most_visited_urls_list = []

    def filter_log(self):

        media_exts = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.gif', '.GIF', '.bmp', '.BMP', '.png', '.PNG', '.mpg', '.MPG']
        with open(OUTPUT_FILE_NAME) as processed_file:
            lines = [next(processed_file) for _ in range(NUMBER_OF_LINES)]
            print("Przed filtracją było", len(lines), "linii.")
            for line in lines:
                if "GET" in line and "200" in line:
                    if any(word in line for word in media_exts):
                        continue
                    else:
                        self.filtered_lines.append(line)

        print("Po filtracji zostało", len(self.filtered_lines), "linii.")

    def extract_values_into_dict(self):

        keys = ['remote_host', 'request_url', 'time_received_datetimeobj', 'request_method', 'request_first_line', ]
        line_parser = apache_log_parser.make_parser("%h %t \"%r\" %s %b")
        for line in self.filtered_lines:
            extracted_data_from_line = line_parser(line)
            filtered_extracted_values = dict(
                (k, extracted_data_from_line[k]) for k in keys if k in extracted_data_from_line)
            self.list_of_extracted_lines.append(filtered_extracted_values)

    def extract_most_visited_urls(self):

        total_count = len(self.filtered_lines)
        occurrences = Counter(k['request_url'] for k in self.list_of_extracted_lines if k.get('request_url'))
        for occurrence, count in occurrences.most_common():
            if count/total_count > 0.005:
                self.most_visited_urls_list.append(occurrence)

        # pprint(self.most_visited_urls_list)
        print("Liczba najczesciej odwiedzanych stron wynosi:", len(self.most_visited_urls_list))
        print("Całkowita liczba odwiedzanych stron wynosi:", total_count)

    def extract_distinct_users(self):

        for elem in self.list_of_extracted_lines:
            if elem['remote_host'] not in self.distinct_user_list:
                self.distinct_user_list.append(elem['remote_host'])
        print("Liczba unikalnych użytkowników wynosi:", len(self.distinct_user_list))

    def extract_sessions(self):

        sessions = {}
        user_session = {}
        offset = timedelta(minutes=30)

        for elem in self.list_of_extracted_lines:
            sessions.setdefault(elem['remote_host'], [])
            sessions[elem['remote_host']].append(elem['time_received_datetimeobj'])
        for user, date_list in sessions.items():
            first_timestamp = datetime(year=MINYEAR, month=1, day=1)
            sessions_count = 0
            cnt = 0
            for date in date_list:
                if date - first_timestamp > offset:
                    if cnt > 1:
                        sessions_count += 1
                    first_timestamp = date
                    cnt = 1
                else:
                    cnt += 1
            if cnt > 1:
                sessions_count += 1
            user_session[user] = sessions_count
        # pprint(user_session)

        total = 0
        for item in user_session.values():
            total += item
        print("Liczba sesji wynosi:", total)


def main():

    parser = Parser()

    parser.filter_log()
    parser.extract_values_into_dict()
    # parser.count_occurrences('remote_host')
    parser.extract_most_visited_urls()
    # parser.count_occurrences('request_url')
    parser.extract_distinct_users()
    parser.extract_sessions()


if __name__ == "__main__":
    main()
