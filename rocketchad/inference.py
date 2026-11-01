def chunked_rollout_forward(forward, rows):
    if rows == 0:
        return forward

    def run(obs):
        count = obs.shape[0]
        if count <= rows:
            return forward(obs)
        output_logits = output_values = None
        for begin in range(0, count, rows):
            end = min(count, begin + rows)
            logits, values = forward(obs[begin:end])
            if output_logits is None:
                output_logits = logits.new_empty((count, logits.shape[1]))
                output_values = values.new_empty((count,))
            output_logits[begin:end].copy_(logits)
            output_values[begin:end].copy_(values)
            del logits, values
        return output_logits, output_values

    return run
