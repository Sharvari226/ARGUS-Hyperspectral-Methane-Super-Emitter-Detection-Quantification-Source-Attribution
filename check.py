from src.db.mongo import get_sync_db
db = get_sync_db()
import time
t = time.time()
r = list(db.detections.find({}, {'_id':0,'cls_embedding':0,'location':0}).limit(5))
print('time:', round(time.time()-t,2), 's')
print('first:', r[0] if r else 'empty')