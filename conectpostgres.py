import hashlib

username = "sisand"
password = "sisand123"
md5_pass = "md5" + hashlib.md5((password + username).encode()).hexdigest()
print(md5_pass)
