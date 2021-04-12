import os
import json
import sys
import datetime

class Output:
  def __init__(self, filename='results.json'):
    self.filename = filename

    with open(self.filename, 'r') as f:
      self.data = json.load(f)
      self.work = {
        'date': datetime.datetime.now().strftime('%m%d_%H_%M'),
        'input': sys.argv[1:],
        'output': []
      }

  def save(self):
    self.data['works'].append(self.work)

    with open(self.filename, 'w') as f:
      json.dump(self.data, f, indent=2)

  def print(self, _string, _value = None, _stdout = True):
    self.work['output'].append([_string, _value])

    if _stdout:
      print(_string, ': ', _value)

if __name__ == '__main__':
  with open('work.txt', 'r') as f:
    for line in f:
      os.system('python main.py ' + line)