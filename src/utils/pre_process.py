import pprint

class PreProcess(object):

    def __init__(self):
        self.query_files = [r"C:\Lucas\PhD\LLMS_multieval\data\2015-subtasks.txt", r"C:\Lucas\PhD\LLMS_multieval\data\2016-subtasks.txt"]
        self.qrels_files = [r"C:\Lucas\PhD\LLMS_multieval\data\qrels-docs-2015.txt", r"C:\Lucas\PhD\LLMS_multieval\data\qrels-docs-2016.txt"]


    def transform_queries_qrels_new_ids(self):
        """
            CALL THIS FUNCTION IN ORDER TO TRANSFORM FROM RAW DATA (TASK TRACK 2015 AND TASK TRACK 2015)
            INTO A NEW DATA WHICH WILL BE STORED IN processed_data
        """
        self.dict_mapper_old_to_new_qid = {}
        self.count_subtasks = {}
        self.dict_missingqueries = {}
        self.dict_assessed_queries = self.get_queries_assessed()
        self.map_old_qid_to_new()
        self.map_qrels_to_newqids()
        exit()
        self.map_old_qid_to_new_withsubtasks()
        self.generate_qrels_subtasks()

    def write_new_query_file(self):
        qid_mapper_file = open(r"C:\Lucas\PhD\LLMS_multieval\data\processed_data\queries\old_qid_to_new.txt", "w")
        qid_query_file = open(r"C:\Lucas\PhD\LLMS_multieval\data\processed_data\queries\qid_query.txt", "w")
        # print("new_qid", "query","year","old_qid",sep="\t")
        print("new_qid", "query","year","old_qid",sep="\t", file=qid_mapper_file)
        print("new_qid", "query",sep="\t", file=qid_query_file)
        for year in ["2015", "2016"]:
            for qid in self.dict_mapper_old_to_new_qid[year]:
                # print(self.dict_mapper_old_to_new_qid[year][qid]['new_qid'], self.dict_mapper_old_to_new_qid[year][qid]['query'], year, qid, sep="\t")
                print(self.dict_mapper_old_to_new_qid[year][qid]['new_qid'], self.dict_mapper_old_to_new_qid[year][qid]['query'], year, qid, sep="\t", file=qid_mapper_file)
                print(self.dict_mapper_old_to_new_qid[year][qid]['new_qid'], self.dict_mapper_old_to_new_qid[year][qid]['query'],sep="\t", file=qid_query_file)

        qid_query_file.close()
        qid_mapper_file.close()

    def map_old_qid_to_new(self):

        new_qid = 1
        for file in self.query_files:
            with open(file) as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]
            if "2015" in file:
                file_label = "2015"
            else:
                file_label = "2016"
            if file_label not in self.dict_mapper_old_to_new_qid:
                self.dict_mapper_old_to_new_qid[file_label] = {}
            for line_number, line in enumerate(content):
                if line.startswith('Task id: '):
                    qid = int(line.split()[2])
                    pprint.pprint(self.dict_assessed_queries)
                    if qid in self.dict_assessed_queries[file_label]:
                        if qid not in self.dict_mapper_old_to_new_qid[file_label]:
                            current_newqid = new_qid
                            if current_newqid not in self.count_subtasks:
                                self.count_subtasks[current_newqid] = 0
                            self.dict_mapper_old_to_new_qid[file_label][qid] = {"new_qid":new_qid,"query":content[line_number+1].split('[')[0].rstrip()}
                            new_qid += 1
                    else:
                        self.dict_missingqueries[qid] = content[line_number+1].split('[')[0].rstrip()
                        current_newqid = None
                elif "*" in line and current_newqid is not None:
                    self.count_subtasks[current_newqid]+= 1

        self.write_new_query_file()

    def get_queries_assessed(self):
        dict_assessed_queries = {}
        for file in self.qrels_files:
            with open(file) as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]
            if "2015" in file:
                file_label = "2015"
            else:
                file_label = "2016"
            if file_label not in dict_assessed_queries:
                dict_assessed_queries[file_label] = {}

            for line in content:
                qid = int(line.split()[0])
                if qid not in dict_assessed_queries[file_label]:
                    dict_assessed_queries[file_label][qid] = ""

        return dict_assessed_queries

    def map_qrels_to_newqids(self):
        qrels_file = open(r"C:\Lucas\PhD\LLMS_multieval\data\processed_data\qrels\qrels_only_query.txt", "w")
        for file in self.qrels_files:
            with open(file) as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]
            if "2015" in file:
                file_label = "2015"
            else:
                file_label = "2016"
            for line in content:
                qid = int(line.split()[0])
                subtask_id = int(line.split()[1])
                if qid in self.dict_mapper_old_to_new_qid[file_label]:
                    if subtask_id == self.count_subtasks[self.dict_mapper_old_to_new_qid[file_label][qid]["new_qid"]]+1:
                        # print(line, "new:",self.dict_mapper_old_to_new_qid[file_label][qid]["new_qid"])
                        print(self.dict_mapper_old_to_new_qid[file_label][qid]["new_qid"],'Q0' ,' '.join(line.split()[2:]),file=qrels_file)
        qrels_file.close()
    def get_new_qid(self, year, qid):
        return self.dict_mapper_old_to_new_qid[year][qid]["new_qid"]

    def get_query_using_old_id(self, year, qid):
        return self.dict_mapper_old_to_new_qid[year][qid]["query"]

    def map_old_qid_to_new_withsubtasks(self):

        self.dict_missingqueries = {}
        self.dict_new_qid_with_subtasks = {}
        new_qid = 1
        for file in self.query_files:
            with open(file, encoding='utf8') as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]
            if "2015" in file:
                file_label = "2015"
            else:
                file_label = "2016"

            for line_number, line in enumerate(content):
                if line.startswith('Task id: '):
                    # print(line)
                    qid = int(line.split()[2])
                    if qid in self.dict_assessed_queries[file_label]:
                        new_qid = self.get_new_qid(file_label,qid)
                        k = line_number + 1
                        if new_qid not in self.dict_new_qid_with_subtasks:
                            self.dict_new_qid_with_subtasks[new_qid] = {"query":self.get_query_using_old_id(file_label,qid),"subtasks":{}}
                            task_id = 1
                            while k < len(content) and not content[k].startswith('Task id: '):
                                if "*" in content[k]:
                                    subtask_query = ' '.join(content[k].rstrip().split()).replace("* ",'')
                                    if task_id not in self.dict_new_qid_with_subtasks[new_qid]["subtasks"]:

                                        self.dict_new_qid_with_subtasks[new_qid]["subtasks"][task_id] = subtask_query
                                    task_id +=1
                                k += 1

        file_with_subtasks = open(r"C:\Lucas\PhD\LLMS_multieval\data\processed_data\queries\qid_subtasks_enumerated.txt", "w")
        pprint.pprint(self.dict_new_qid_with_subtasks, file_with_subtasks)
        self.write_query_with_subtasks_file()

    def write_query_with_subtasks_file(self):
        qid_subtasks_file = open(r"C:\Lucas\PhD\LLMS_multieval\data\processed_data\queries\qid_with_subtasks.txt", "w")
        # print("new_qid", "query","year","old_qid",sep="\t")
        print("new_qid", "query", sep="\t", file=qid_subtasks_file)
        for new_qid, subtasks in sorted(self.dict_new_qid_with_subtasks.items()):
            print(str(new_qid)+"_0", subtasks['query'],sep="\t", file=qid_subtasks_file)
            for subtask_id, subtask in subtasks['subtasks'].items():
                print(str(new_qid)+"_"+str(subtask_id), subtask, sep="\t", file=qid_subtasks_file)

        qid_subtasks_file.close()

    def generate_qrels_subtasks(self):
        qrels_file = open(r"C:\Lucas\PhD\LLMS_multieval\data\processed_data\qrels\qrels_merged_withsubtasks.txt", "w")
        for file in self.qrels_files:
            with open(file) as f:
                content = f.readlines()
            content = [x.rstrip() for x in content]
            if "2015" in file:
                file_label = "2015"
            else:
                file_label = "2016"
            for line in content:
                qid = int(line.split()[0])
                subtask_id = int(line.split()[1])
                new_qid = int(self.get_new_qid(file_label, qid))
                if subtask_id == self.count_subtasks[new_qid] + 1:
                    print(str(new_qid)+"_0", 'Q0',' '.join(line.split()[2:]), file=qrels_file)
                else:
                    print(str(new_qid) + "_"+str(subtask_id), 'Q0', ' '.join(line.split()[2:]),file=qrels_file)
        qrels_file.close()


if __name__ == '__main__':
    preprocess = PreProcess()
    preprocess.transform_queries_qrels_new_ids()

