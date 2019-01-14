#!/usr/bin/env python

import sys
sysargv = sys.argv[:]
sys.argv = sys.argv[:1]

from googleapiclient.discovery import build, MediaFileUpload
from httplib2 import Http
from oauth2client import file, client, tools

import hashlib
import json
import os
import requests


REPONAME = "[DASH][BIN]"

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive'
TOKEN = None


def get_service():
    global TOKEN
    store = file.Storage('token.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))
    return service


def get_folder():
    service = get_service()
    results = service.files().list(
        q="name contains '{}'".format(REPONAME),
        pageSize=10, fields="nextPageToken, files(id, name, modifiedTime)").execute()
    items = results.get('files', [])
    assert len(items) == 1
    return items[0]['id']


def list_drive():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    service = get_service()
    parent = get_folder()
    results = service.files().list(
        q="'{}' in parents and trashed=false".format(parent),
        pageSize=10, fields="nextPageToken, files(id, name, modifiedTime)").execute()
    items = results.get('files', [])
    return items


def list_files():
    try:
        with open(".hashes", "r") as fname:
            gitstatus = json.load(fname)
    except FileNotFoundError:
        os.mknod(".hashes")
        with open(".hashes", "w") as fname:
            fname.write("[]")
        gitstatus = []
    try:
        fnames = os.listdir('./storage')
    except FileNotFoundError:
        os.mkdir("./storage")
        fnames = []
    staged, unstaged = [], []
    for fileobj in gitstatus:
        if fileobj['name'] in fnames:
            with open("./storage/" + fileobj['name'], "rb") as fname:
                hasher = hashlib.sha512()
                hasher.update(fname.read())
                hash = hasher.digest().hex()
                if fileobj['hash'] == hash:
                    staged.append(fileobj)
                else:
                    unstaged.append(fileobj)
    for fname in fnames:
        if fname not in [x['name'] for x in staged]:
            with open("./storage/" + fname, "rb") as fileobj:
                hasher = hashlib.sha512()
                hasher.update(fileobj.read())
                hash = hasher.digest().hex()
            unstaged.append({"name": fname, "hash": hash})
    return staged, unstaged


def diff(staged, items):
    unpulled, really_staged = [], []
    for fstaged in staged:
        same_id = [x for x in items if x['id'] == fstaged['id']]
        if same_id:
            if same_id[0]['modifiedTime'] != fstaged['modifiedTime']:
                unpulled.append(same_id[0])
        else:
            really_staged.append(fstaged)
    return unpulled, really_staged


def push(fname):
    parent = get_folder()
    service = get_service()
    metadata = {
        "name": fname.split("/")[-1],
        "parents": [parent],
        'mimeType': None
    }
    media = MediaFileUpload(fname,
                            mimetype=None,
                            resumable=True)
    try:
        meta = service.files().create(body=metadata,
                                      media_body=media,
                                      fields='id, modifiedTime').execute()
        return meta
    except Exception:
        print("File {} is probably empty".format(fname))


if __name__ == '__main__':
    if len(sysargv) == 1:
        task = "status"
    elif sysargv[1] == "status":
        task = "status"
    elif sysargv[1] == "pull":
        task = "pull"
    elif sysargv[1] == "push":
        task = "push"
    else:
        print("FATAL: incorrect task [status, pull, push]")
        quit()
    items = list_drive()
    staged, unstaged = list_files()
    unpulled, staged = diff(staged, items)
    if task == "status":
        print("Drive status")
        print("Unpulled")
        for fileobj in unpulled:
            print(">  ", fileobj["name"])
        print("Staged")
        for fileobj in staged:
            print(">  ", fileobj["name"])
        print("Unstaged")
        for fileobj in unstaged:
            print(">  ", fileobj["name"])
    elif task == "pull":
        print("Drive pulling...")
    elif task == "push":
        print("Drive pushing...")
        for fileobj in unstaged:
            meta = push("./storage/" + fileobj["name"])
            if meta:
                fileobj.update(meta)
                staged.append(fileobj)
        with open(".hashes", "w") as f:
            json.dump(staged, f)
