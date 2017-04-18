#!/usr/bin/env python
# Load experimental and model data
import os, sqlite3, json
from hmax.levels import util


class Data:
    def __init__(self):
        self.workerIds = set()

    def load_participant_json(self, experiment_run, verbose=True):
        exp_filename = util.get_experiment_db_filename_by_run(experiment_run)
        assert(os.path.isfile(exp_filename))
        con = sqlite3.connect(exp_filename)
        cur = con.cursor()
        cur.execute(
            "SELECT workerid,beginhit,status,datastring FROM placecat WHERE status in (3,4) AND NOT datastring==''")
        data = cur.fetchall()
        if verbose: print "%d participants found in file %s." % (len(data), exp_filename)
        con.close()
        return data

    def load_participant_ids(self, experiment_run):
        r = self.load_participant_json(experiment_run, verbose=True)
        for i_subj in range(0,len(r)):
            self.workerIds.add(json.loads(r[i_subj][3])['workerId'])

partiticpants_table_sql = '''
    CREATE TABLE placecat (
        uniqueid VARCHAR(128) NOT NULL,
        assignmentid VARCHAR(128) NOT NULL,
        workerid VARCHAR(128) NOT NULL,
        hitid VARCHAR(128) NOT NULL,
        ipaddress VARCHAR(128),
        browser VARCHAR(128),
        platform VARCHAR(128),
        language VARCHAR(128),
        cond INTEGER,
        counterbalance INTEGER,
        codeversion VARCHAR(128),
        beginhit DATETIME,
        beginexp DATETIME,
        endhit DATETIME,
        bonus FLOAT,
        status INTEGER,
        datastring TEXT(4294967295),
        PRIMARY KEY (uniqueid)
    );'''

def save_dummy_participant_db(repeat_participants, output_filename):
    # Delete previous DB
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    output_db = sqlite3.connect(output_filename)
    cur = output_db.cursor()
    cur.execute(partiticpants_table_sql)
    # Insert repeat participants
    q = "INSERT INTO placecat (workerid, uniqueid, assignmentid, hitid) VALUES ('%s',%s,'%s','%s')"
    for idx, par in enumerate(repeat_participants):
        cur.execute(q % (par, idx, idx, idx))
    output_db.commit()
    output_db.close()


def create_dummy_participant_db(experiment_names, output_filename):
    """
        Create a new particpants database with dummy entries for all participants who did the experiments
        in list experiment_names
    """
    data = Data()
    for exp_name in experiment_names:
        data.load_participant_ids(experiment_run=exp_name)

    dup_workers = data.workerIds
    if len(dup_workers):
        print 'Found repeat workers: %s' % dup_workers
    else:
        print 'No repeat workers!'
    save_dummy_participant_db(dup_workers, output_filename)
