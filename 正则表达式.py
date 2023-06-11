import re
# string = 'please sendswitch=1okplease send'
string = 'a123b'
t = re.findall(r"a(.+?)b", string)
print(t)