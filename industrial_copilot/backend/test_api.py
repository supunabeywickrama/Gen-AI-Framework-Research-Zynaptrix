import requests
import json

data = {
    'machine_id': 'TEST-123',
    'name': 'test',
    'location': 'loc',
    'manual_id': 'manul',
    'sensors': json.dumps([{"sensor_id": "test1", "sensor_name": "test1"}])
}

try:
    res = requests.post('http://127.0.0.1:8000/api/machines', data=data)
    with open('test_res.txt', 'w') as f:
        f.write(str(res.status_code) + "\n" + res.text)
except Exception as e:
    print(e)
