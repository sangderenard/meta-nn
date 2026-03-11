# meta-nn

`meta-nn` is the extracted training and orchestration workspace for a graph-shaped meta network: a set of cooperating nodes that build vocabulary, prepare data possessions, train staged models, gate progress, and feed later stages with the results of earlier ones.

The system is designed around explicit graph execution rather than a single monolithic training loop. Each node owns one concern, edges define when data is provisioned, and gate nodes decide whether downstream stages are ready to run.

## Design Overview

The pipeline is organized as layered graph views. The Mermaid blocks below are generated from `pipeline/graph_layers.py` so the README stays aligned with the actual graph definitions. Each generated section now includes a dense infographic plus collapsible alternate views for simpler topology and reaction-focused inspection.

Run `python update_readme_graphs.py` after changing graph layers.

### Training Execution Layer

<!-- BEGIN:GENERATED_EXECUTION_LAYER -->
**Execution layer Dense Infographic**

```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart TD
    subgraph group_bootstrap[Bootstrap]
        wave_pool["Wave Pool<br/>object / bootstrap / seed"]
    end
    subgraph group_build[Build]
        build_classifier["Build Classifier<br/>object / build / builder"]
        config_search["Config Search<br/>object / build / node"]
        build_transformer["Build Transformer<br/>object / build / builder"]
        build_gan["Build GAN<br/>object / build / builder"]
        build_wave_classifier["Build Wave Classifier<br/>object / build / builder"]
    end
    subgraph group_vocab[Vocab]
        init_vocab["Init Vocab<br/>object / vocab / seed"]
        vocab_churn["Vocab Churn<br/>object / vocab / seed"]
        build_symbol_pool["Build Symbol Pool<br/>object / vocab / builder"]
        build_label_embedding["Build Label Embedding<br/>object / vocab / builder"]
        build_flashcard_rows["Build Flashcard Rows<br/>object / vocab / builder"]
    end
    subgraph group_data[Data]
        data_node["Data Authority<br/>object / data / storage_authority"]
    end
    subgraph group_train[Train]
        stage_0_pregestation["Pregestation Train<br/>object / train / stage"]
        stage_1_gestation["Gestation Train<br/>object / train / stage"]
        stage_2_berkeley["Berkeley Refresh Train<br/>object / train / stage"]
        stage_r_transformer["Transformer Train<br/>object / train / stage"]
        stage_g_generator["Generator Train<br/>object / train / stage"]
        stage_w_wave_classifier["Wave Classifier Train<br/>object / train / stage"]
        stage_c_lora["LoRA Round<br/>object / train / stage"]
        stage_fake_feedback["Fake Class Feedback<br/>object / train / stage"]
    end
    subgraph group_gates[Gates]
        gate_berkeley["Berkeley Gate<br/>object / gates / gate"]
        gate_transformer["Transformer Gate<br/>object / gates / gate"]
        gate_generator["Generator Gate<br/>object / gates / gate"]
        gate_wave["Wave Gate<br/>object / gates / gate"]
    end
    subgraph group_housekeeping[Housekeeping]
        sync_gate_replica["Sync Gate Replica<br/>object / housekeeping / housekeeping"]
        checkpoint_save["Checkpoint Save<br/>object / housekeeping / housekeeping"]
    end

    wave_pool -- "startup" --> init_vocab
    init_vocab -- "startup" --> build_classifier
    build_classifier -- "startup" --> config_search
    config_search -- "startup" --> build_transformer
    build_transformer -- "if_gan_mode" --> build_gan
    wave_pool -- "startup" --> build_wave_classifier
    build_classifier -- "per_round" --> vocab_churn
    vocab_churn -- "per_round" --> build_symbol_pool
    build_symbol_pool -- "per_round" --> build_label_embedding
    build_label_embedding -- "per_round" --> data_node
    data_node -- "provides:pregestation_loader" --> stage_0_pregestation
    data_node -- "provides:gestation_loader" --> stage_1_gestation
    data_node -- "provides:berkeley_refresh_loader" --> stage_2_berkeley
    data_node -- "provides:gate_val_loader+payload_val_loader" --> gate_berkeley
    data_node -- "provides:payload_bank" --> stage_g_generator
    data_node -- "provides:payload_images" --> build_flashcard_rows
    stage_2_berkeley -- "after_gate1" --> gate_berkeley
    gate_berkeley -- "after_gate1" --> stage_r_transformer
    stage_r_transformer -- "after_gate1" --> gate_transformer
    gate_transformer -- "after_all_gates" --> stage_g_generator
    stage_g_generator -- "after_all_gates" --> gate_generator
    gate_berkeley -- "after_all_gates" --> stage_c_lora
    stage_c_lora -- "after_all_gates" --> stage_fake_feedback
    build_wave_classifier -- "after_transformer_gate" --> stage_w_wave_classifier
    stage_w_wave_classifier -- "after_transformer_gate" --> gate_wave
    gate_wave -- "end_of_round" --> sync_gate_replica
    gate_transformer -- "end_of_round" --> sync_gate_replica
    gate_berkeley -- "end_of_round" --> sync_gate_replica
    sync_gate_replica -- "end_of_round" --> checkpoint_save

    classDef faculty_bootstrap fill:#E9F1F7,stroke:#4B6B88,color:#102A43,stroke-width:2px;
    classDef faculty_build fill:#F8EFE5,stroke:#B07219,color:#40210F,stroke-width:2px;
    classDef faculty_vocab fill:#FFF7CC,stroke:#9A7D0A,color:#3D3100,stroke-width:2px;
    classDef faculty_data fill:#DFF6F5,stroke:#127475,color:#053B3C,stroke-width:2px;
    classDef faculty_train fill:#FFE8D6,stroke:#C05621,color:#4A1D05,stroke-width:2px;
    classDef faculty_gates fill:#FDE2E4,stroke:#C0392B,color:#4A0F13,stroke-width:2px;
    classDef faculty_housekeeping fill:#E8F5E9,stroke:#2E7D32,color:#102A12,stroke-width:2px;
    classDef faculty_io fill:#DDEBFF,stroke:#2563EB,color:#0F172A,stroke-width:2px;
    classDef faculty_inference fill:#F4F1DE,stroke:#3D405B,color:#1B1F2A,stroke-width:2px;
    classDef faculty_buffer fill:#E0FBFC,stroke:#006D77,color:#00313A,stroke-width:2px;
    classDef faculty_gate fill:#FDE2E4,stroke:#C0392B,color:#4A0F13,stroke-width:2px;
    classDef faculty_other fill:#F3F4F6,stroke:#6B7280,color:#111827,stroke-width:2px;
    class wave_pool faculty_bootstrap;
    class init_vocab,vocab_churn,build_symbol_pool,build_label_embedding,build_flashcard_rows faculty_vocab;
    class build_classifier,config_search,build_transformer,build_gan,build_wave_classifier faculty_build;
    class data_node faculty_data;
    class stage_0_pregestation,stage_1_gestation,stage_2_berkeley,stage_r_transformer,stage_g_generator,stage_w_wave_classifier,stage_c_lora,stage_fake_feedback faculty_train;
    class gate_berkeley,gate_transformer,gate_generator,gate_wave faculty_gates;
    class sync_gate_replica,checkpoint_save faculty_housekeeping;
    linkStyle 0 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 1 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 2 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 3 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 4 stroke:#577590,stroke-width:3px,opacity:0.85,stroke-dasharray:8 3;
    linkStyle 5 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 6 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 7 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 8 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 9 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 10 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 11 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 12 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 13 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 14 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 15 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 16 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 17 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 18 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 19 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 20 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 21 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 22 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 23 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 24 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 25 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 26 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 27 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 28 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
```

Node colors group bootstrap, build, vocab, data, train, gate, and housekeeping faculties. Edge colors separate startup, per-round, data-provision, gated progression, and end-of-round reactions.

<details>

<summary>Minimal schematic view</summary>



```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart TD
    wave_pool["Wave Pool"]
    init_vocab["Init Vocab"]
    build_classifier["Build Classifier"]
    config_search["Config Search"]
    build_transformer["Build Transformer"]
    build_gan["Build GAN"]
    build_wave_classifier["Build Wave Classifier"]
    vocab_churn["Vocab Churn"]
    build_symbol_pool["Build Symbol Pool"]
    build_label_embedding["Build Label Embedding"]
    build_flashcard_rows["Build Flashcard Rows"]
    data_node["Data Authority"]
    stage_0_pregestation["Pregestation Train"]
    stage_1_gestation["Gestation Train"]
    stage_2_berkeley["Berkeley Refresh Train"]
    stage_r_transformer["Transformer Train"]
    stage_g_generator["Generator Train"]
    stage_w_wave_classifier["Wave Classifier Train"]
    stage_c_lora["LoRA Round"]
    stage_fake_feedback["Fake Class Feedback"]
    gate_berkeley["Berkeley Gate"]
    gate_transformer["Transformer Gate"]
    gate_generator["Generator Gate"]
    gate_wave["Wave Gate"]
    sync_gate_replica["Sync Gate Replica"]
    checkpoint_save["Checkpoint Save"]

    wave_pool --> init_vocab
    init_vocab --> build_classifier
    build_classifier --> config_search
    config_search --> build_transformer
    build_transformer --> build_gan
    wave_pool --> build_wave_classifier
    build_classifier --> vocab_churn
    vocab_churn --> build_symbol_pool
    build_symbol_pool --> build_label_embedding
    build_label_embedding --> data_node
    data_node --> stage_0_pregestation
    data_node --> stage_1_gestation
    data_node --> stage_2_berkeley
    data_node --> gate_berkeley
    data_node --> stage_g_generator
    data_node --> build_flashcard_rows
    stage_2_berkeley --> gate_berkeley
    gate_berkeley --> stage_r_transformer
    stage_r_transformer --> gate_transformer
    gate_transformer --> stage_g_generator
    stage_g_generator --> gate_generator
    gate_berkeley --> stage_c_lora
    stage_c_lora --> stage_fake_feedback
    build_wave_classifier --> stage_w_wave_classifier
    stage_w_wave_classifier --> gate_wave
    gate_wave --> sync_gate_replica
    gate_transformer --> sync_gate_replica
    gate_berkeley --> sync_gate_replica
    sync_gate_replica --> checkpoint_save
```

</details>

<details>

<summary>Reaction-colored view</summary>



```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart TD
    wave_pool["Wave Pool"]
    init_vocab["Init Vocab"]
    build_classifier["Build Classifier"]
    config_search["Config Search"]
    build_transformer["Build Transformer"]
    build_gan["Build GAN"]
    build_wave_classifier["Build Wave Classifier"]
    vocab_churn["Vocab Churn"]
    build_symbol_pool["Build Symbol Pool"]
    build_label_embedding["Build Label Embedding"]
    build_flashcard_rows["Build Flashcard Rows"]
    data_node["Data Authority"]
    stage_0_pregestation["Pregestation Train"]
    stage_1_gestation["Gestation Train"]
    stage_2_berkeley["Berkeley Refresh Train"]
    stage_r_transformer["Transformer Train"]
    stage_g_generator["Generator Train"]
    stage_w_wave_classifier["Wave Classifier Train"]
    stage_c_lora["LoRA Round"]
    stage_fake_feedback["Fake Class Feedback"]
    gate_berkeley["Berkeley Gate"]
    gate_transformer["Transformer Gate"]
    gate_generator["Generator Gate"]
    gate_wave["Wave Gate"]
    sync_gate_replica["Sync Gate Replica"]
    checkpoint_save["Checkpoint Save"]

    wave_pool -- "startup | schedule.bootstrap" --> init_vocab
    init_vocab -- "startup | schedule.bootstrap" --> build_classifier
    build_classifier -- "startup | schedule.bootstrap" --> config_search
    config_search -- "startup | schedule.bootstrap" --> build_transformer
    build_transformer -- "if_gan_mode | schedule.transition" --> build_gan
    wave_pool -- "startup | schedule.bootstrap" --> build_wave_classifier
    build_classifier -- "per_round | schedule.round" --> vocab_churn
    vocab_churn -- "per_round | schedule.round" --> build_symbol_pool
    build_symbol_pool -- "per_round | schedule.round" --> build_label_embedding
    build_label_embedding -- "per_round | schedule.round" --> data_node
    data_node -- "provides:pregestation_loader | data.provide" --> stage_0_pregestation
    data_node -- "provides:gestation_loader | data.provide" --> stage_1_gestation
    data_node -- "provides:berkeley_refresh_loader | data.provide" --> stage_2_berkeley
    data_node -- "provides:gate_val_loader+payload_val_loader | data.provide" --> gate_berkeley
    data_node -- "provides:payload_bank | data.provide" --> stage_g_generator
    data_node -- "provides:payload_images | data.provide" --> build_flashcard_rows
    stage_2_berkeley -- "after_gate1 | schedule.transition" --> gate_berkeley
    gate_berkeley -- "after_gate1 | schedule.transition" --> stage_r_transformer
    stage_r_transformer -- "after_gate1 | schedule.transition" --> gate_transformer
    gate_transformer -- "after_all_gates | schedule.transition" --> stage_g_generator
    stage_g_generator -- "after_all_gates | schedule.transition" --> gate_generator
    gate_berkeley -- "after_all_gates | schedule.transition" --> stage_c_lora
    stage_c_lora -- "after_all_gates | schedule.transition" --> stage_fake_feedback
    build_wave_classifier -- "after_transformer_gate | schedule.transition" --> stage_w_wave_classifier
    stage_w_wave_classifier -- "after_transformer_gate | schedule.transition" --> gate_wave
    gate_wave -- "end_of_round | schedule.finalize" --> sync_gate_replica
    gate_transformer -- "end_of_round | schedule.finalize" --> sync_gate_replica
    gate_berkeley -- "end_of_round | schedule.finalize" --> sync_gate_replica
    sync_gate_replica -- "end_of_round | schedule.finalize" --> checkpoint_save

    linkStyle 0 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 1 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 2 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 3 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 4 stroke:#577590,stroke-width:3px,opacity:0.85,stroke-dasharray:8 3;
    linkStyle 5 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 6 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 7 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 8 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 9 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 10 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 11 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 12 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 13 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 14 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 15 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 16 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 17 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 18 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 19 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 20 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 21 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 22 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 23 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 24 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 25 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 26 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 27 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 28 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
```

</details>
<!-- END:GENERATED_EXECUTION_LAYER -->

### Inference Edge Layer

<!-- BEGIN:GENERATED_INFERENCE_LAYER -->
**Inference layer Dense Infographic**

```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    subgraph group_io[Io]
        input_bus[["Input Bus<br/>bus / io / input_bus"]]
        output_bus[["Output Bus<br/>bus / io / output_bus"]]
    end
    subgraph group_inference[Inference]
        wave_classifier_runtime["Wave Classifier<br/>object / inference / wave_classifier"]
        transformer_runtime["Transformer<br/>object / inference / transformer"]
        classifier_runtime["Classifier<br/>object / inference / classifier"]
        generator_runtime["GAN Generator<br/>object / inference / generator"]
    end
    subgraph group_buffer[Buffer]
        wave_sidecar_extract["Wave Sidecar Extract<br/>object / buffer / wave_sidecar_extractor"]
        wave_repack["Wave Repack<br/>object / buffer / wave_repacker"]
    end
    subgraph group_gate[Gate]
        discriminator_runtime["Discriminator<br/>object / gate / discriminator"]
    end

    input_bus -- "ingress wave" --> wave_classifier_runtime
    wave_classifier_runtime -- "bitplane window" --> transformer_runtime
    transformer_runtime -- "carrier sidecar" --> wave_sidecar_extract
    transformer_runtime -- "transformed bitplane" --> classifier_runtime
    classifier_runtime -- "semantic condition" --> generator_runtime
    generator_runtime -- "candidate synthesis" --> discriminator_runtime
    discriminator_runtime -- "reject and retry" --> generator_runtime
    discriminator_runtime -- "accepted candidate" --> wave_repack
    wave_sidecar_extract -- "preserved wave surround" --> wave_repack
    transformer_runtime -- "bitplane pre/post transform" --> wave_repack
    wave_repack -- "egress wave" --> output_bus

    classDef faculty_bootstrap fill:#E9F1F7,stroke:#4B6B88,color:#102A43,stroke-width:2px;
    classDef faculty_build fill:#F8EFE5,stroke:#B07219,color:#40210F,stroke-width:2px;
    classDef faculty_vocab fill:#FFF7CC,stroke:#9A7D0A,color:#3D3100,stroke-width:2px;
    classDef faculty_data fill:#DFF6F5,stroke:#127475,color:#053B3C,stroke-width:2px;
    classDef faculty_train fill:#FFE8D6,stroke:#C05621,color:#4A1D05,stroke-width:2px;
    classDef faculty_gates fill:#FDE2E4,stroke:#C0392B,color:#4A0F13,stroke-width:2px;
    classDef faculty_housekeeping fill:#E8F5E9,stroke:#2E7D32,color:#102A12,stroke-width:2px;
    classDef faculty_io fill:#DDEBFF,stroke:#2563EB,color:#0F172A,stroke-width:2px;
    classDef faculty_inference fill:#F4F1DE,stroke:#3D405B,color:#1B1F2A,stroke-width:2px;
    classDef faculty_buffer fill:#E0FBFC,stroke:#006D77,color:#00313A,stroke-width:2px;
    classDef faculty_gate fill:#FDE2E4,stroke:#C0392B,color:#4A0F13,stroke-width:2px;
    classDef faculty_other fill:#F3F4F6,stroke:#6B7280,color:#111827,stroke-width:2px;
    class input_bus,output_bus faculty_io;
    class wave_classifier_runtime,transformer_runtime,classifier_runtime,generator_runtime faculty_inference;
    class wave_sidecar_extract,wave_repack faculty_buffer;
    class discriminator_runtime faculty_gate;
    linkStyle 0 stroke:#0077B6,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 1 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 2 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 3 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 4 stroke:#264653,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 5 stroke:#E76F51,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 6 stroke:#D62828,stroke-width:4px,opacity:0.95,stroke-dasharray:8 4;
    linkStyle 7 stroke:#2B9348,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 8 stroke:#577590,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 9 stroke:#7F5539,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 10 stroke:#3A86FF,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
```

Node colors group buses, inference objects, buffers, and acceptance gates. Edge colors separate ingress, interpretation, synthesis, retry-loop, repack, and egress paths.

<details>

<summary>Minimal schematic view</summary>



```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    input_bus[["Input Bus"]]
    wave_classifier_runtime["Wave Classifier"]
    transformer_runtime["Transformer"]
    wave_sidecar_extract["Wave Sidecar Extract"]
    classifier_runtime["Classifier"]
    generator_runtime["GAN Generator"]
    discriminator_runtime["Discriminator"]
    wave_repack["Wave Repack"]
    output_bus[["Output Bus"]]

    input_bus --> wave_classifier_runtime
    wave_classifier_runtime --> transformer_runtime
    transformer_runtime --> wave_sidecar_extract
    transformer_runtime --> classifier_runtime
    classifier_runtime --> generator_runtime
    generator_runtime --> discriminator_runtime
    discriminator_runtime --> generator_runtime
    discriminator_runtime --> wave_repack
    wave_sidecar_extract --> wave_repack
    transformer_runtime --> wave_repack
    wave_repack --> output_bus
```

</details>

<details>

<summary>Reaction-colored view</summary>



```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    input_bus[["Input Bus"]]
    wave_classifier_runtime["Wave Classifier"]
    transformer_runtime["Transformer"]
    wave_sidecar_extract["Wave Sidecar Extract"]
    classifier_runtime["Classifier"]
    generator_runtime["GAN Generator"]
    discriminator_runtime["Discriminator"]
    wave_repack["Wave Repack"]
    output_bus[["Output Bus"]]

    input_bus -- "ingress wave | bus.read" --> wave_classifier_runtime
    wave_classifier_runtime -- "bitplane window | infer.classify_wave" --> transformer_runtime
    transformer_runtime -- "carrier sidecar | wave.extract_sidecar" --> wave_sidecar_extract
    transformer_runtime -- "transformed bitplane | infer.transform" --> classifier_runtime
    classifier_runtime -- "semantic condition | infer.condition" --> generator_runtime
    generator_runtime -- "candidate synthesis | infer.generate" --> discriminator_runtime
    discriminator_runtime -- "reject and retry | infer.retry_loop" --> generator_runtime
    discriminator_runtime -- "accepted candidate | infer.accept" --> wave_repack
    wave_sidecar_extract -- "preserved wave surround | wave.sidecar_attach" --> wave_repack
    transformer_runtime -- "bitplane pre/post transform | wave.repack" --> wave_repack
    wave_repack -- "egress wave | bus.write" --> output_bus

    linkStyle 0 stroke:#0077B6,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 1 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 2 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 3 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 4 stroke:#264653,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 5 stroke:#E76F51,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 6 stroke:#D62828,stroke-width:4px,opacity:0.95,stroke-dasharray:8 4;
    linkStyle 7 stroke:#2B9348,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 8 stroke:#577590,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 9 stroke:#7F5539,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 10 stroke:#3A86FF,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
```

</details>
<!-- END:GENERATED_INFERENCE_LAYER -->

## Node Roles

- `InitVocabNode`: establishes the current term inventory and semantic target space.
- `VocabChurnNode`: mutates or refreshes vocabulary state on the configured churn schedule.
- `BuildSymbolPoolNode`: prepares symbol/image sources used by gestation-stage data building.
- `BuildLabelEmbeddingNode`: creates the embedding basis used for semantic conditioning and scoring.
- `DataNode`: the storage authority for wave pools, pregestation rows, gestation rows, Berkeley rows, and payloads; data is provisioned across edges on demand.
- `PregestationTrainNode`: trains the earliest synthetic stage on diversified pregestation passes.
- `GestationTrainNode`: advances training using symbol-pool and mixed-source gestation data.
- `BerkeleyRefreshTrainNode`: trains the Berkeley semantic stage against full-vocabulary image/mask data.
- `BerkeleyGateNode`: decides whether the semantic stage is strong enough to unlock later training.
- `ConfigSearchNode`: searches runtime/render/training configuration for viable operating points.
- `BuildTransformerNode`: materializes the transformer once configuration is accepted.
- `TransformerTrainNode`: learns sequence or feature transformations used by later stages.
- `TransformerGateNode`: evaluates whether transformer outputs justify proceeding.
- `BuildGANNode`: instantiates the conditional generator/discriminator pair.
- `GeneratorTrainNode`: trains the generator through adversarial and classifier-mediated semantic feedback.
- `GeneratorGateNode`: checks whether generator quality is good enough to continue.
- `LoRARoundNode`: applies parameter-efficient adaptation rounds against the current classifier state.
- `FakeClassFeedbackNode`: feeds generated samples back through the classifier path as semantic pressure.
- `BuildWaveClassifierNode`: constructs the waveform/image classifier used in the wave stage.
- `WaveClassifierTrainNode`: trains the wave classifier from rendered waveform records and labels.
- `WaveGateNode`: decides whether wave-side readiness criteria are met.
- `SyncGateReplicaNode`: synchronizes gate state and replica state for the next round.
- `CheckpointSaveNode`: persists runtime state, model state, and graph progress.
- `BuildFlashcardRowsNode`: emits payload rows and derivative examples for downstream inspection or study artifacts.

## Interaction Model

The key architectural rule is that nodes do not reach into one another ad hoc. Data is supplied when an edge is traversed. That means:

- topology lives in the orchestrator
- storage policy lives in `DataNode`
- training logic lives in stage nodes
- progression policy lives in gate nodes

This gives the meta network a clear separation between:

- what exists
- when it is built
- who consumes it
- whether later stages are allowed to run

## Layered Graph Intent

The repo now treats the current training graph as the **execution layer** of a broader graph system.
That execution layer is intentionally narrow:

- runtime nodes are modeled as objects with a faculty and an archetype
- runtime edges carry a target function plus a default parameter dictionary that can merge runtime overrides
- the edge list remains the scheduling surface, while edge reactions remain the execution surface

This leaves room for adjacent graph layers without overloading the execution graph:

- `execution`: the graph that actually runs now
- `inference`: runtime bus-oriented inference edges through the trained model objects
- `provenance`: future ownership, storage-location, and lineage edges for data possessions
- `stack_view`: future Nodus-facing operation stacks, buffer handoffs, and dependency-timed tick moments

The immediate goal is not to make `meta-nn` run like Nodus. The goal is to keep the execution layer fluent and simple now while making sure other graph paradigms can be projected from the same runtime conception later.

## Repository Layout

- `pipeline/`: graph model, context, orchestrator, node implementations, protocol definitions, and utilities.
- `wav_ml_core.py`: waveform record, rendering, and basic audio utility functions.
- `wav_ml_models.py`: model definitions, training helpers, and runtime configuration helpers.
- `semantic_dataset_loaders.py`: Berkeley semantic data loading, degradation, mask construction, and dataset support.
- `wav_pipeline_graph.py`: graph-oriented entrypoint bridging the older monolith into the extracted pipeline structure.

## Current Intent

This repository is the isolated home for the meta-network effort. The immediate goal is to keep the graph-native training system moving independently of the original mixed C++ and Python workspace, so the network architecture, staged training flow, and data-node design can evolve on their own boundary.
