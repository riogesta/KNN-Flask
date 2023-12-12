import yaml

def check_k_value():
    # membaca konfigurasi berkas YAML
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    return config.get('KNN', {}).get('K')

def write_k_value(k):
    # membaca konfigurasi berkas YAML
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    # memperbaharui nilai K pada konfigurasi
    config['KNN']['K'] = k
    
    # menuliskan kembali konfigurasi ke berkas YAML
    with open('config.yaml', 'w') as config_file:
        yaml.dump(config, config_file, default_flow_style=False)