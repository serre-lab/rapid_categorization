#!/usr/bin/env python

import os
import shutil
import sqlite3
from rapid_categorization.config import psiturk_run_path
from rapid_categorization.model import util


class merger:
    def __init__(self):
        self.db_con = []
        self.output_db = []
        self.participants_table_sql = '''
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
        self.overwrite = True

    def db_merge(self, child):
        child_db = util.get_experiment_db_filename_by_run(child)
        self.db_con.execute("ATTACH '%s' as dba" % child_db)
        self.db_con.execute("BEGIN")
        for row in self.db_con.execute(
                "SELECT * FROM dba.sqlite_master WHERE type='table'"):
            combine = "INSERT OR IGNORE INTO %s SELECT * FROM dba.%s" % (
                row[1], row[1])
            self.db_con.execute(combine)
        self.output_db.commit()
        self.db_con.execute("detach database dba")

    def combine_dbs(self, dbs, output_name):
        run_path = os.path.join(psiturk_run_path, output_name)
        if self.overwrite:
            if os.path.isfile(run_path):
                os.remove(run_path)
        else:
            if os.path.isfile(run_path):
                print 'Combined DB already exists. Exiting code.'
            return
        self.output_db = sqlite3.connect(run_path)
        self.db_con = self.output_db.cursor()
        self.db_con.execute(self.participants_table_sql)
        [self.db_merge(x) for x in dbs]
        self.output_db.close()
        target_path = util.get_experiment_db_filename_by_run(
            output_name)
        shutil.move(run_path, target_path)
        print 'Saved combined db to: %s' % run_path


if __name__ == '__main__':
    merge_dbs = merger()
    merge_dbs.combine_dbs(
        # dbs=[
        #     'click_center_probfill_400stim_150res',
        #     'click_center_probfill_400stim_150res_2',
        #     'click_center_probfill_400stim_150res_3',
        #     'click_center_probfill_400stim_150res_4',
        #     'click_center_probfill_400stim_150res_5'
        #     ],
        # output_name='click_center_probfill_400stim_150res_combined')



        dbs=[
            'lrp_center_probfill_400stim_150res',
            'lrp_center_probfill_400stim_150res_2',
            ],
        output_name='lrp_center_probfill_400stim_150res_combined')
