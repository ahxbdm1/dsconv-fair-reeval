"""Verify retained commit-pinned evidence; do not redistribute downloaded code."""
import argparse,csv,hashlib,json,pathlib,collections
ROOT=pathlib.Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--online',action='store_true',help='Fetch evidence files into memory and verify whole-file digests and line ranges')
args=parser.parse_args()
def read(name):return list(csv.DictReader((ROOT/'audit'/name).open(encoding='utf-8-sig',newline='')))
protocol=read('audit_protocol_codex.csv');ports=read('audit_dsconv_ports_codex.csv');evidence=read('audit_evidence_codex.csv')
ids={x['evidence_id'] for x in evidence};assert len(ids)==len(evidence)
assert all(set(x['evidence_ids'].split(';'))<=ids for x in protocol+ports)
verified={}
if args.online:
    import requests
    for row in evidence:
        key=(row['repository'],row['commit'],row['path'])
        if key not in verified:
            response=requests.get(f'https://raw.githubusercontent.com/{key[0]}/{key[1]}/{key[2]}',timeout=60)
            response.raise_for_status();raw=response.content
            assert hashlib.sha256(raw).hexdigest()==row['sha256'],key
            verified[key]=len(raw.decode('utf-8-sig').splitlines())
        # The table preserves the manual classification and exact line range;
        # automated digest verification does not replace scientific judgment.
        assert 1<=int(row['line_start'])<=int(row['line_end'])<=verified[key]
summary={'protocol_rows':len(protocol),'eligible_vessel_paths':sum(r['eligible_vessel_training'].lower()=='true' for r in protocol),'port_rows':len(ports),'port_classification':dict(collections.Counter(r['classification'] for r in ports)),'evidence_rows':len(evidence),'whole_source_files_verified_online':len(verified),'limitation':'Convenience sample; code copies are correlated. No community prevalence or private execution inference.'}
print(json.dumps(summary,indent=2))
