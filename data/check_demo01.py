import h5py

file_path = 'expert_data/demo01.hdf5'
print(f'Checking file: {file_path}')

try:
    with h5py.File(file_path, 'r') as f:
        print(f'\nKeys in root: {list(f.keys())}')
        
        if 'data' in f:
            print(f'\nKeys in data group: {list(f["data"].keys())}')
            
            # Check for demo groups
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            print(f'\nFound {len(demo_keys)} demonstration(s):')
            
            for demo_key in demo_keys:
                demo_group = f['data'][demo_key]
                print(f'\n{demo_key}:')
                print(f'  Datasets: {list(demo_group.keys())}')
                print(f'  Attributes: {dict(demo_group.attrs)}')
                
                # Check shapes
                if 'states' in demo_group:
                    print(f'  states shape: {demo_group["states"].shape}')
                if 'actions' in demo_group:
                    print(f'  actions shape: {demo_group["actions"].shape}')
                if 'initial_state' in demo_group:
                    print(f'  initial_state shape: {demo_group["initial_state"].shape}')
            
            # Check data attributes
            print(f'\nData group attributes: {dict(f["data"].attrs)}')
        else:
            print('ERROR: No data group found!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
