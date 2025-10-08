from typing import List
import torch
import torch.nn as nn
import json
import os
from .tokenizer import Tokenizer
from . import LLM

from fairscale.nn.model_parallel import initialize as fs_init


class MetaModel(nn.Module):

    def __init__(self, llama_type, llama_config, llama_ckpt_dir=None, tokenizer_path=None,
                 capture_layers=False, output_dir=None):
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        ModelArgs = LLM.__dict__[llama_type].ModelArgs
        Transformer = LLM.__dict__[llama_type].Transformer

        with open(llama_config, "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=2048, max_batch_size=32, **params
        )
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words

        model = Transformer(model_args)
        mp_rank = fs_init.get_model_parallel_rank()
        if llama_ckpt_dir is not None:
            ckpt_path = os.path.join(llama_ckpt_dir, f"consolidated.{mp_rank:02d}.pth")
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location="cpu")
                msg = model.load_state_dict(checkpoint, strict=False)
                print(msg)
            else:
                print(f'Checkpoint not found at {ckpt_path}')
        self.llma = model

        # Initialize layer capture functionality
        self.capture_layers = capture_layers
        if self.capture_layers:
            try:
                from ..util.layer_capture import LayerOutputCapture
            except ImportError:
                from util.layer_capture import LayerOutputCapture
            self.layer_capture = LayerOutputCapture(
                output_dir=output_dir or "./layer_outputs",
                enabled=True
            )
            print(f"Layer capture enabled, saving to: {self.layer_capture.output_dir}")
        else:
            self.layer_capture = None

        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameter count : {count}")

    def forward(self, examples, labels, image=None, modal='image'):
        output = self.llma(examples, image=image, modal=modal, layer_capture=self.layer_capture)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            c_loss = self.criterion(output.reshape(-1, self.tokenizer.n_words), labels.flatten())

        return c_loss

    def generate(
        self,
        prompts: List[str],
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
    ) -> List[str]:
        bsz = len(prompts)
        params = self.llma.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full(
            (bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal=modal)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
    
    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        images,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        modal = ['image'],
    ):
        params = self.llma.params

        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        # truncate from the left. leave some space for generation.
        max_seq_len = params.max_seq_len
        if images is not None:
            max_seq_len -= self.llma.image_words

        max_prompt_size = max_seq_len - max_gen_len
        prompt_tokens = prompt_tokens[-max_prompt_size:]

        prompt_size = len(prompt_tokens)

        total_len = min(max_seq_len, max_gen_len + prompt_size)

        tokens = torch.full([total_len], 0).cuda().long()

        tokens[:len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
        start_pos = prompt_size
        prev_pos = 0
        generate_until = start_pos
        for cur_pos in range(start_pos, total_len):
            logits = self.llma.forward_inference(tokens[None, prev_pos:cur_pos], prev_pos, images if prev_pos == 0 else None, modal = modal)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = self.sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.item()

            if next_token == self.tokenizer.eos_id:
                break

            tokens[cur_pos] = next_token
            prev_pos = cur_pos
            generate_until = cur_pos + 1
            yield {"text": self.tokenizer.decode(tokens[start_pos:generate_until].tolist()), "end_of_content": False}

        yield {"text": self.tokenizer.decode(tokens[start_pos:generate_until].tolist()), "end_of_content": True}

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def get_image_words(self):
        return self.llma.image_words

    def inference_image_only(self, sample_id: str, image, modal='image'):
        """
        Run inference with image only input and capture layer outputs.

        Args:
            sample_id: Unique identifier for this sample
            image: Input image tensor
            modal: Modality type ('image', 'video', etc.)

        Returns:
            dict: Captured layer outputs information
        """
        if not self.capture_layers:
            raise ValueError("Layer capture must be enabled during model initialization")

        # Set up sample for capture
        self.layer_capture.set_sample(
            sample_id=sample_id,
            metadata={
                "input_type": "image_only",
                "modal": modal,
                "model_type": "onellm"
            }
        )

        # Create dummy text (empty caption)
        dummy_text = ""
        prompt_tokens = [self.tokenizer.encode(dummy_text, bos=True, eos=False)]

        # Create tensor with just BOS token
        tokens = torch.full((1, 1), self.tokenizer.bos_id, dtype=torch.long).cuda()

        # Run forward pass with image
        with torch.no_grad():
            try:
                # Use the transformer directly for image-only processing
                h = self.llma.tok_embeddings(tokens)
                h_bos = h[:, :1]

                # Process image
                image_tokens = self.llma.encode_image(image, modal=modal, layer_capture=self.layer_capture)

                # Combine: BOS + start_tag + image_tokens + end_tag
                h_combined = torch.cat((
                    h_bos,
                    self.llma.start_tag[modal].expand(1, -1, -1),
                    image_tokens,
                    self.llma.end_tag[modal].expand(1, -1, -1)
                ), dim=1)

                # Run through LLaMA layers
                self.llma.freqs_cis = self.llma.freqs_cis.to(h_combined.device)
                seqlen = h_combined.shape[1]
                freqs_cis = self.llma.freqs_cis[0:seqlen]
                mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h_combined.device)
                mask = torch.triu(mask, diagonal=1).type_as(h_combined)

                # Capture all LLaMA layers
                for i, layer in enumerate(self.llma.layers):
                    h_combined = layer(h_combined, 0, freqs_cis, mask)

                    if self.layer_capture.enabled:
                        self.layer_capture.capture_layer(
                            f"llama_layer_{i+1:02d}",
                            h_combined,
                            additional_info={
                                "layer_number": i + 1,
                                "layer_type": "transformer_block",
                                "input_type": "image_only",
                                "sequence_length": h_combined.shape[1],
                                "feature_dim": h_combined.shape[2]
                            }
                        )

                # Finish capture
                self.layer_capture.finish_sample()

                return {
                    "status": "success",
                    "sample_id": sample_id,
                    "sequence_length": seqlen,
                    "layers_captured": 32 + 1  # pre_llama + 32 llama layers
                }

            except Exception as e:
                self.layer_capture.finish_sample()
                return {
                    "status": "error",
                    "sample_id": sample_id,
                    "error": str(e)
                }

    def inference_text_only(self, sample_id: str, text):
        """
        Run inference with text only input and capture layer outputs.

        Args:
            sample_id: Unique identifier for this sample
            text: Input text string

        Returns:
            dict: Captured layer outputs information
        """
        if not self.capture_layers:
            raise ValueError("Layer capture must be enabled during model initialization")

        # Set up sample for capture
        self.layer_capture.set_sample(
            sample_id=sample_id,
            metadata={
                "input_type": "text_only",
                "text": text,
                "model_type": "onellm"
            }
        )

        # Tokenize text
        prompt_tokens = [self.tokenizer.encode(text, bos=True, eos=False)]
        max_len = max(len(t) for t in prompt_tokens)

        # Create tensor
        tokens = torch.full((1, max_len), self.tokenizer.pad_id, dtype=torch.long).cuda()
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t).long()

        # Run forward pass
        with torch.no_grad():
            try:
                # Use the transformer directly for text-only processing
                output = self.llma(tokens, image=None, modal=['image'], layer_capture=self.layer_capture)

                # Finish capture
                self.layer_capture.finish_sample()

                return {
                    "status": "success",
                    "sample_id": sample_id,
                    "sequence_length": max_len,
                    "text": text,
                    "layers_captured": 1 + 32  # embedding + 32 llama layers
                }

            except Exception as e:
                self.layer_capture.finish_sample()
                return {
                    "status": "error",
                    "sample_id": sample_id,
                    "error": str(e)
                }