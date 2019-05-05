import csv
import apache_log_parser
from pprint import pprint
from collections import Counter
from datetime import timedelta


INPUT_FILE_NAME = "log.txt"
OUTPUT_FILE_NAME = "parsed_log.txt"
NUMBER_OF_LINES = 50000


def preprocess_log():
    """ Helper function, takes input file and extract first 50000 lines into new file. """
    with open(INPUT_FILE_NAME, 'r') as input_file:
        with open(OUTPUT_FILE_NAME, 'w') as output_file:
            for _ in range(NUMBER_OF_LINES):
                output_file.write(input_file.readline())


class Data(object):
    def __init__(self, host_name, session=None):
        self.host_name = host_name
        self.session = session
        self.request_url = []
        self.time_received_datetimeobj = []
        self.request_method = []
        self.request_first_line = []
        self.session_time = None

    def add(self, *args):
        request_url = args[0] if len(args) >= 1 else None
        time_received_datetimeobj = args[1] if len(args) >= 2 else None
        request_method = args[2] if len(args) >= 3 else None
        request_first_line = args[3] if len(args) >= 4 else None
        self.request_url.append(request_url)
        self.time_received_datetimeobj.append(time_received_datetimeobj)
        self.request_method.append(request_method)
        self.request_first_line.append(request_first_line)

    def get(self, i):
        return (self.request_url[i],
                self.time_received_datetimeobj[i],
                self.request_method[i],
                self.request_first_line[i])

    def __str__(self):
        if self.session is not None:
            string = "{}:sesja={}\n{}\n{}\n{}\n{}\n".format(self.host_name,
                                                            self.session,
                                                            self.request_url,
                                                            self.time_received_datetimeobj,
                                                            self.request_method,
                                                            self.request_first_line)
        else:
            string = "{}\n{}\n{}\n{}\n{}\n".format(self.host_name,
                                                   self.request_url,
                                                   self.time_received_datetimeobj,
                                                   self.request_method,
                                                   self.request_first_line)
        return string

    def __len__(self):
        return len(self.request_url)


class Parser(object):

    def __init__(self):
        self.filtered_lines = []
        self.copy_filtered_lines = []
        self.list_of_extracted_lines = []
        self.distinct_user_list = []
        self.most_visited_urls_list = []
        self.unique_hosts = {}
        self.unique_sessions = {}
        self.unique_users_with_urls = {}

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
                        self.copy_filtered_lines.append(line)

        print("Po filtracji zostało", len(self.filtered_lines), "linii.")

    def extract_values_into_dict(self):
        keys = ['remote_host', 'request_url']
        line_parser = apache_log_parser.make_parser("%h %t \"%r\" %s %b")
        for line in self.copy_filtered_lines:
            extracted_data_from_line = line_parser(line)
            filtered_extracted_values = dict(
                (k, extracted_data_from_line[k]) for k in keys if k in extracted_data_from_line)
            self.list_of_extracted_lines.append(filtered_extracted_values)

    def extract_values_into_data(self):
        keys = ['request_url', 'time_received_datetimeobj', 'request_method', 'request_first_line', ]
        line_parser = apache_log_parser.make_parser("%h %t \"%r\" %s %b")
        for line in self.filtered_lines:
            extracted_data_from_line = line_parser(line)
            host_name = extracted_data_from_line["remote_host"]
            self.unique_hosts.setdefault(host_name, Data(host_name))
            self.unique_hosts[host_name].add(
                extracted_data_from_line[keys[0]],
                extracted_data_from_line[keys[1]],
                extracted_data_from_line[keys[2]],
                extracted_data_from_line[keys[3]],
            )

    def extract_most_visited_urls(self):
        total_count = len(self.filtered_lines)
        occurrences = Counter(k['request_url'] for k in self.list_of_extracted_lines if k.get('request_url'))
        for occurrence, count in occurrences.most_common():
            if count / total_count > 0.005:
                # print(occurrence,  '{:.2f}%'.format(count/total_count*100))
                self.most_visited_urls_list.append(occurrence)

        # print("@ATTRIBUTE", 'session_time', 'NUMERIC')
        # print("@ATTRIBUTE", 'avg_session_time', 'NUMERIC')
        # print("@ATTRIBUTE", 'visited_pages', 'NUMERIC')

        # for elem in self.most_visited_urls_list:
        #     print("@ATTRIBUTE", elem, "{F,T}")

        # print("@DATA")

        # print("Liczba najczesciej odwiedzanych stron wynosi:", len(self.most_visited_urls_list))
        # print("Całkowita liczba odwiedzanych stron wynosi:", total_count)

    def extract_sessions(self):
        offset = timedelta(minutes=30)
        unique_sessions = {}

        for host in self.unique_hosts.values():
            url, first_timestamp, req, line = host.get(0)
            unique_user = host.host_name + '_0'
            unique_sessions[unique_user] = Data(host.host_name, 0)
            unique_sessions[unique_user].add(url, first_timestamp, req, line)
            counter = 0
            for i in range(1, len(host)):
                url, time, req, line = host.get(i)
                if time - first_timestamp > offset:
                    counter += 1
                    unique_user = host.host_name + '_' + str(counter)
                    unique_sessions[unique_user] = Data(host.host_name, counter)
                    unique_sessions[unique_user].add(url, time, req, line)
                    first_timestamp = time
                    # print(unique_sessions[unique_user])

                else:
                    unique_sessions[unique_user].add(url, time, req, line)

        for k, v in unique_sessions.items():
            if len(v) > 1:
                self.unique_sessions[k] = v

        # print("Liczba użytkowników wynosi", len(self.unique_hosts))
        # print("Liczba sesji wynosi ", len(self.unique_sessions))

    def get_avg_session_time(self):
        for session in self.unique_sessions.values():
            if len(session) <= 1:
                session.avg_session_time = timedelta(0)
                continue
            session.avg_session_time = (session.time_received_datetimeobj[-1] - session.time_received_datetimeobj[
                0]) / (len(session))

    def get_session_time(self):
        for session in self.unique_sessions.values():
            if len(session) <= 1:
                session.session_time = timedelta(0)
                continue
            session.session_time = (session.time_received_datetimeobj[-1] - session.time_received_datetimeobj[0])

    def make_session_analyze(self):
        matrix = [[] for _ in range(len(self.unique_sessions))]

        for j, session in enumerate(self.unique_sessions.values()):
            if 0 <= session.session_time.total_seconds() < 200:
                session.session_time = "short"
            elif 200 <= session.session_time.total_seconds() < 800:
                session.session_time = "medium"
            elif 800 <= session.session_time.total_seconds():
                session.session_time = "long"
            matrix[j].append(session.session_time)
            if 0 <= session.avg_session_time.total_seconds() < 50:
                session.avg_session_time = "short"
            elif 50 <= session.avg_session_time.total_seconds() < 200:
                session.avg_session_time = "medium"
            elif 200 <= session.avg_session_time.total_seconds():
                session.avg_session_time = "long"
            matrix[j].append(session.avg_session_time)
            if 0 <= len(session) < 3:
                matrix[j].append("few")
            elif 3 <= len(session) < 8:
                matrix[j].append("medium")
            elif 8 <= len(session):
                matrix[j].append("many")
            # for url in self.most_visited_urls_list:
            #     matrix[j].append("T" if url in session.request_url else "F")

        with open('input_session_analyze_file.csv', mode='w') as input_file:
            input_writer = csv.writer(input_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in matrix:
                input_writer.writerow(i)

    def make_user_analyze(self):
        matrix = [[] for _ in range(len(self.unique_hosts))]

        for i, host in enumerate(self.unique_hosts.values()):
            urls = set(host.request_url)
            for url in self.most_visited_urls_list:
                matrix[i].append('T' if url in urls else "F")

        with open('input_user_analyze_file.csv', mode='w') as input_file:
            input_writer = csv.writer(input_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for i in matrix:
                input_writer.writerow(i)
                # print(i)


def main():
    parser = Parser()

    parser.filter_log()
    parser.extract_values_into_data()
    parser.extract_values_into_dict()
    parser.extract_most_visited_urls()
    parser.extract_sessions()
    parser.get_avg_session_time()
    parser.get_session_time()
    parser.make_session_analyze()
    parser.make_user_analyze()


if __name__ == "__main__":
    main()
