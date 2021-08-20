from os import listdir
from os.path import isfile, join


class RunsTrack(object):
    def __init__(self):
        self.runs_track_folder = r"data/runs_task_track/"
        self.runs_track_newids = r"data/runs_track_newids/"
        self.map_old_qid_new_file = "data/processed_data/queries/old_qid_to_new.txt"
        self.dict_old_qids = self.read_dict_mapper_qids()

    def get_list_files(self, path):
        onlyfiles = [path+f for f in listdir(path) if isfile(join(path, f))]
        return onlyfiles

    def read_dict_mapper_qids(self):
        dict_old_qids = {}
        with open(self.map_old_qid_new_file) as f:
            content = f.readlines()
        content = [x.rstrip() for x in content]
        for line in content[1:]:
            parts = line.split('\t')
            qid = parts[0]
            query = parts[1]
            year = parts[2]
            old_qid = parts[3]
            if year not in dict_old_qids:
                dict_old_qids[year] = {}
            if old_qid not in dict_old_qids[year]:
                dict_old_qids[year][old_qid] = qid

        return dict_old_qids

    def get_new_qid(self, year, qid):
        try:
            return self.dict_old_qids[year][qid]
        except Exception as inst:
            return None


    def generate_runs_new_ids(self):
        for year in ['2015/','2016/']:
            runs_by_year = self.get_list_files(self.runs_track_folder+year)
            for run in runs_by_year:
                with open(run) as f:
                    content = f.readlines()
                content = [x.rstrip() for x in content]
                filename = run.split('/')[-1]
                file_out = open(self.runs_track_newids+year+filename, 'w')

                for line in content:
                    parts = line.split()
                    new_qid = None
                    new_qid = self.get_new_qid(year.replace('/',''), parts[0])
                    if new_qid is not None:
                        parts[0] = new_qid
                        print(' '.join(parts), file=file_out)
                file_out.close()