"""Atomic session boundary commits: model, optimizer, RNG and four banks agree."""
from pathlib import Path
import json
import os
import shutil
from common import atomic, read, file_hash, digest

def latest(output,fingerprint):
    output=Path(output);pointer=output/'latest.json'
    if not pointer.exists():return None
    p=read(pointer);directory=output/'checkpoints'/p['directory']
    if directory.parent != output/'checkpoints':raise ValueError('Invalid checkpoint pointer')
    manifest=read(directory/'COMMITTED.json')
    if manifest['fingerprint']!=fingerprint or digest(manifest)!=p['manifest_digest']:
        raise RuntimeError('Checkpoint identity mismatch')
    for rel,h in manifest['files'].items():
        if file_hash(directory/rel)!=h:raise RuntimeError('Incomplete/corrupt checkpoint '+rel)
    return directory,read(directory/'state.json')

def publish(output,sequence,fingerprint,state,save_payload):
    """Rank zero only; save_payload writes adapter, optimizer and RNG into staging."""
    output=Path(output);base=output/'checkpoints';base.mkdir(parents=True,exist_ok=True)
    previous=read(output/'latest.json') if (output/'latest.json').exists() else None
    name=f'session_{sequence:06d}';pending=base/(name+'.pending');dest=base/name
    if pending.exists():shutil.rmtree(pending)
    pending.mkdir();save_payload(pending);atomic(pending/'state.json',state)
    # torch.save/save_pretrained close files but do not guarantee durable writes.
    for p in pending.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    files={str(p.relative_to(pending)):file_hash(p) for p in pending.rglob('*') if p.is_file()}
    manifest={'fingerprint':fingerprint,'sequence':sequence,'files':files}
    atomic(pending/'COMMITTED.json',manifest)
    # A complete orphan from a crash before pointer publication can be replaced safely.
    if dest.exists():
        if previous and previous['directory']==name:raise RuntimeError('Refusing to overwrite current checkpoint')
        shutil.rmtree(dest)
    os.replace(pending,dest)
    fd=os.open(base,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    atomic(output/'latest.json',{'directory':name,'manifest_digest':digest(manifest)})
    # The new pointer is durable before rotation; never delete original SFT or traces.
    if previous and previous['directory']!=name:
        old=base/previous['directory']
        if old.parent==base and old.exists():shutil.rmtree(old)
    return dest
