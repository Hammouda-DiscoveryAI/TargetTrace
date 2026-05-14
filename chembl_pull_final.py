import os
import glob
import time
import requests
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR    = "protein_data_chunks"
FINAL_FILE    = "ml_ready_protein_dataset.csv"
ID_CACHE      = "target_ids_cache.txt"
MIN_BINDERS   = 50
UNIPROT_DELAY = 0.2
PAGE_SIZE     = 200   # Records per page — small enough to be reliable
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_uniprot_accession(target_obj):
    components = target_obj.get('target_components', [])
    if not components:
        return None
    for comp in components:
        acc = comp.get('accession')
        if acc:
            return acc
        for syn in comp.get('component_synonyms', []):
            if syn.get('syn_type') == 'UNIPROT':
                return syn.get('component_synonym')
    return None


def fetch_sequence_from_uniprot(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        lines = resp.text.strip().split('\n')
        sequence = ''.join(lines[1:])
        return sequence if sequence else None
    except Exception:
        return None


def get_all_single_protein_target_ids():
    """
    Paginates through ALL ChEMBL single-protein targets using explicit
    offset slicing on a FRESH queryset each page. This is the only reliable
    way to guarantee all records are fetched — reusing the same queryset
    object causes the client to silently stop early.

    .order_by('target_chembl_id') ensures stable ordering so pages don't
    overlap or skip records between requests.

    Saves IDs to disk every page. On restart loads from cache instantly.
    Delete target_ids_cache.txt to force a fresh fetch.
    """
    # RESUME: load from cache if previous run completed this step
    if os.path.exists(ID_CACHE):
        with open(ID_CACHE, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(ids)} target IDs from cache ({ID_CACHE}).")
        return ids

    print(f"Fetching all single-protein target IDs from ChEMBL...")
    print(f"Saving to {ID_CACHE} every {PAGE_SIZE} records (safe to restart)\n")

    target_api = new_client.target
    ids = []
    seen = set()   # Dedup guard in case pages overlap
    offset = 0
    consecutive_empty = 0

    with open(ID_CACHE, 'w') as cache_file:
        with tqdm(desc="Fetching target IDs", unit=" targets") as pbar:
            while True:
                try:
                    # CRITICAL: build a completely fresh queryset every iteration
                    page = list(
                        target_api
                        .filter(target_type='SINGLE PROTEIN')
                        .only(['target_chembl_id'])
                        .order_by('target_chembl_id')[offset:offset + PAGE_SIZE]
                    )
                except Exception as e:
                    tqdm.write(f"[Warning] Page at offset {offset} failed: {e}. Retrying in 5s...")
                    time.sleep(5)
                    continue

                if not page:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        # Three empty pages in a row — we've reached the end
                        break
                    offset += PAGE_SIZE
                    continue

                consecutive_empty = 0
                new_on_page = 0

                for t in page:
                    tid = t.get('target_chembl_id')
                    if tid and tid not in seen:
                        seen.add(tid)
                        ids.append(tid)
                        cache_file.write(tid + '\n')
                        new_on_page += 1

                cache_file.flush()
                pbar.update(new_on_page)
                offset += PAGE_SIZE

                # If this page returned fewer records than PAGE_SIZE, we're done
                if len(page) < PAGE_SIZE:
                    break

    print(f"\nFetched {len(ids)} target IDs. Saved to {ID_CACHE}.")
    return ids


def fetch_and_save_activities(target_id, pbar, counters):
    """
    For one target:
      1. Fetches full record from ChEMBL via .get()
      2. Extracts UniProt accession
      3. Fetches sequence from UniProt REST API
      4. Fetches all IC50/Kd activities
      5. Writes CSV immediately if data passes all filters

    One target in RAM at a time. Writes to disk before moving on.
    Skips entirely if CSV already exists (resume feature).
    """
    file_path = os.path.join(OUTPUT_DIR, f"{target_id}.csv")

    def update_bar():
        pbar.set_postfix(ordered_dict={
            'saved'   : counters['saved'],
            'too_few' : counters['too_few'],
            'no_acc'  : counters['no_accession'],
            'no_seq'  : counters['no_sequence'],
            'resumed' : counters['resumed'],
            'errors'  : counters['errors'],
        }, refresh=True)

    # RESUME: skip if already saved
    if os.path.exists(file_path):
        counters['resumed'] += 1
        update_bar()
        return

    try:
        target_obj = new_client.target.get(target_id)
        if not target_obj:
            counters['errors'] += 1
            update_bar()
            return

        target_name = target_obj.get('pref_name', 'Unknown Protein')

        accession = get_uniprot_accession(target_obj)
        if not accession:
            counters['no_accession'] += 1
            update_bar()
            return

        time.sleep(UNIPROT_DELAY)
        sequence = fetch_sequence_from_uniprot(accession)
        if not sequence:
            counters['no_sequence'] += 1
            update_bar()
            return

        # Fetch all activities — fully evaluated, no lazy querysets
        activities = list(
            new_client.activity.filter(
                target_chembl_id=target_id,
                standard_type__in=['IC50', 'Kd'],
                standard_units='nM'
            )
        )

        if len(activities) < MIN_BINDERS:
            counters['too_few'] += 1
            update_bar()
            return

        data_list = []
        for act in activities:
            smiles   = act.get('canonical_smiles')
            value    = act.get('standard_value')
            relation = act.get('standard_relation', '')
            if not smiles or not value:
                continue
            if relation not in ('=', '~'):
                continue
            data_list.append({
                'target_chembl_id' : target_id,
                'uniprot_accession': accession,
                'protein_name'     : target_name,
                'protein_sequence' : sequence,
                'binder_id'        : act.get('molecule_chembl_id'),
                'smiles'           : smiles,
                'property_type'    : act.get('standard_type'),
                'potency_value_nm' : value,
                'relation'         : relation,
            })

        if len(data_list) < MIN_BINDERS:
            counters['too_few'] += 1
            update_bar()
            return

        # Write immediately — safe to crash after this line
        df = pd.DataFrame(data_list)
        df.to_csv(file_path, index=False)
        del df

        counters['saved'] += 1
        pbar.write(f"[Saved] {target_id} | {target_name} | {len(data_list)} rows")
        update_bar()

    except Exception as e:
        counters['errors'] += 1
        pbar.write(f"[Error] {target_id}: {str(e)}")
        update_bar()


def combine_datasets(directory, output_name):
    """
    Stream-merges all chunk CSVs into one master file.
    Reads one file at a time — full dataset never in RAM simultaneously.
    """
    print("\n--- Starting Data Consolidation ---")
    all_files = glob.glob(os.path.join(directory, "*.csv"))

    if not all_files:
        print("No valid data files found.")
        return

    print(f"Stream-merging {len(all_files)} protein chunk files...")

    write_header     = True
    rows_written     = 0
    proteins_written = 0

    for f in tqdm(all_files, desc="Merging chunks"):
        try:
            df = pd.read_csv(f)
            df.dropna(subset=['protein_sequence', 'smiles', 'potency_value_nm'], inplace=True)
            df['potency_value_nm'] = pd.to_numeric(df['potency_value_nm'], errors='coerce')
            df.dropna(subset=['potency_value_nm'], inplace=True)
            if df.empty:
                continue
            df.to_csv(output_name, mode='a', header=write_header, index=False)
            write_header = False
            rows_written += len(df)
            proteins_written += 1
            del df
        except Exception as e:
            print(f"[Warning] Could not read {f}: {e}")

    if rows_written == 0:
        print("No valid rows found.")
        return

    print(f"\nSuccess! Master dataset saved to: {output_name}")
    print(f"  Total unique proteins : {proteins_written}")
    print(f"  Total bioactivity rows: {rows_written}")


def sanity_check():
    print("--- Running sanity check on CHEMBL203 (EGFR) ---")
    try:
        test_acts = list(
            new_client.activity.filter(
                target_chembl_id='CHEMBL203',
                standard_type__in=['IC50', 'Kd'],
                standard_units='nM'
            )[:3]
        )
        if not test_acts:
            print("[FAILED] No activities returned from ChEMBL.")
            return False
        smiles = test_acts[0].get('canonical_smiles')
        if not smiles:
            print("[FAILED] canonical_smiles missing.")
            return False
        print(f"[OK] ChEMBL activity   : {smiles[:50]}...")

        target_obj = new_client.target.get('CHEMBL203')
        accession = get_uniprot_accession(target_obj)
        if not accession:
            print("[FAILED] Could not extract UniProt accession.")
            return False
        print(f"[OK] UniProt accession : {accession}")

        sequence = fetch_sequence_from_uniprot(accession)
        if not sequence:
            print(f"[FAILED] Could not fetch sequence from UniProt.")
            return False
        print(f"[OK] Protein sequence  : {sequence[:40]}...")

        # Verify pagination works — fetch page 0 and page 1, confirm no overlap
        target_api = new_client.target
        page0 = list(
            target_api.filter(target_type='SINGLE PROTEIN')
            .only(['target_chembl_id'])
            .order_by('target_chembl_id')[0:5]
        )
        page1 = list(
            target_api.filter(target_type='SINGLE PROTEIN')
            .only(['target_chembl_id'])
            .order_by('target_chembl_id')[5:10]
        )
        ids0 = {t['target_chembl_id'] for t in page0}
        ids1 = {t['target_chembl_id'] for t in page1}
        overlap = ids0 & ids1
        if overlap:
            print(f"[FAILED] Pagination overlap detected: {overlap}")
            return False
        if not page0 or not page1:
            print("[FAILED] Pagination returned empty pages.")
            return False
        print(f"[OK] Pagination        : page0={sorted(ids0)[:2]}... page1={sorted(ids1)[:2]}... no overlap")

        print("[Sanity Check PASSED]\n")
        return True

    except Exception as e:
        print(f"[Sanity Check FAILED] {str(e)}")
        return False


if __name__ == "__main__":
    if not sanity_check():
        print("Aborting.")
        exit(1)

    # Step 1: fetch all target IDs with live progress, saved page by page
    # Delete target_ids_cache.txt to force a fresh fetch
    target_ids = get_all_single_protein_target_ids()

    # Step 2: per-target — fetch sequence + activities, write CSV immediately
    counters = {
        'saved'       : 0,
        'resumed'     : 0,
        'no_accession': 0,
        'no_sequence' : 0,
        'too_few'     : 0,
        'errors'      : 0,
    }

    print(f"\nProcessing {len(target_ids)} targets (MIN_BINDERS={MIN_BINDERS})...")
    with tqdm(total=len(target_ids), desc="Downloading proteins", unit=" target") as pbar:
        for target_id in target_ids:
            fetch_and_save_activities(target_id, pbar, counters)
            pbar.update(1)

    print(f"\n--- Run Summary ---")
    print(f"  Saved               : {counters['saved']}")
    print(f"  Resumed (skipped)   : {counters['resumed']}")
    print(f"  No UniProt accession: {counters['no_accession']}")
    print(f"  No sequence         : {counters['no_sequence']}")
    print(f"  Too few binders     : {counters['too_few']}")
    print(f"  Errors              : {counters['errors']}")

    # Step 3: stream-merge all chunks into final ML-ready dataset
    combine_datasets(OUTPUT_DIR, FINAL_FILE)
