tf_config = {
        'cluster': {
            'worker' : ['192.168.0.236:30000', '192.168.0.236:30001'],
            #'worker' : ['192.168.0.236:30000'],
        },
        'task': {'type': 'worker'}
}