#!/bin/sh
db=$1
sqlite3 /ssd/db/${db} "attach \"/ssd/db/BaseDataMap.db\" as db1; insert into sid2db (sid, dbname) select sid, dbname from info;" 
