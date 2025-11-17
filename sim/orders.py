import simpy

def order_manager_process(env, store, orders, num_pickers):
    """
    Legger ordre i køen i SimPy og avslutter når ingen ordre gjenstår.
    """
    # legg inn ordrer
    for i, order_list in enumerate(orders):
        store.put({
            "order_id": f"O{i+1:03d}",
            "order_list": order_list
        })
        yield env.timeout(0)

    # send STOP-signaler—ett per plukker
    for _ in range(num_pickers):
        store.put({
            "order_id": "STOP",
            "order_list": None
        })
        yield env.timeout(0)
