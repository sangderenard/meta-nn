# meta-nn

`meta-nn` is the extracted training and orchestration workspace for a graph-shaped meta network: a set of cooperating nodes that build vocabulary, prepare data possessions, train staged models, gate progress, and feed later stages with the results of earlier ones.

The system is designed around explicit graph execution rather than a single monolithic training loop. Each node owns one concern, edges define when data is provisioned, and gate nodes decide whether downstream stages are ready to run.

## Design Overview

The pipeline is organized as layered graph views. The Mermaid blocks below are generated from `pipeline/graph_layers.py` so the README stays aligned with the actual graph definitions. Each generated section now includes a dense infographic plus collapsible alternate views for simpler topology and reaction-focused inspection.

Run `python update_readme_graphs.py` after changing graph layers.

### Training Execution Layer

<!-- BEGIN:GENERATED_EXECUTION_LAYER -->
**Training Composite Execution Overlay**

```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart TD
    subgraph group_bootstrap[Bootstrap]
        wave_pool["Wave Pool<br/>object / bootstrap / seed  [once]"]
    end
    subgraph group_build[Build]
        build_classifier["Build Classifier<br/>object / build / builder  [once]"]
        config_search["Config Search<br/>object / build / node  [once]"]
        build_transformer["Build Transformer<br/>object / build / builder  [once]"]
        build_gan["Build GAN<br/>object / build / builder  [once]"]
        build_wave_classifier["Build Wave Classifier<br/>object / build / builder"]
    end
    subgraph group_vocab[Vocab]
        init_vocab["Init Vocab<br/>object / vocab / seed  [once]"]
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
        stage_1_gestation["Gestation Train<br/>object / train / stage  [gated]"]
        stage_2_berkeley["Berkeley Refresh Train<br/>object / train / stage  [gated]"]
        stage_r_transformer["Transformer Train<br/>object / train / stage  [gated]"]
        stage_g_generator["Generator Train<br/>object / train / stage  [gated]"]
        stage_w_wave_classifier["Wave Classifier Train<br/>object / train / stage  [gated]"]
        stage_c_lora["LoRA Round<br/>object / train / stage  [gated]"]
        stage_fake_feedback["Fake Class Feedback<br/>object / train / stage  [gated]"]
    end
    subgraph group_gates[Gates]
        gate_0_pregestation_eval["Pregestation Eval<br/>object / gates / gate"]
        gate_1_gestation_eval["Gestation Eval<br/>object / gates / gate  [gated]"]
        gate_berkeley["Berkeley Gate<br/>object / gates / gate  [gated]"]
        gate_transformer["Transformer Gate<br/>object / gates / gate  [gated]"]
        gate_generator["Generator Gate<br/>object / gates / gate  [gated]"]
        gate_wave["Wave Gate<br/>object / gates / gate  [gated]"]
        decision_004_stage_w_wave_classifier{"Wave stage ready?<br/>control / gates / decision"}
        decision_007_gate_wave{"Wave stage ready?<br/>control / gates / decision"}
        decision_010_build_gan{"GAN mode?<br/>control / gates / decision"}
        decision_016_stage_1_gestation{"Gate 0 passed?<br/>control / gates / decision"}
        decision_017_gate_1_gestation_eval{"Gate 0 passed?<br/>control / gates / decision"}
        decision_018_stage_2_berkeley{"Gate 1 passed?<br/>control / gates / decision"}
        decision_019_gate_berkeley{"Gate 1 passed?<br/>control / gates / decision"}
        decision_020_stage_r_transformer{"Gate 1 passed?<br/>control / gates / decision"}
        decision_021_stage_c_lora{"Gate 2 passed?<br/>control / gates / decision"}
        decision_022_gate_transformer{"Gate 1 passed?<br/>control / gates / decision"}
        decision_023_stage_fake_feedback{"Gate 2 passed?<br/>control / gates / decision"}
        decision_024_stage_g_generator{"Gate 2 passed?<br/>control / gates / decision"}
        decision_026_gate_generator{"Gate 2 passed?<br/>control / gates / decision"}
    end
    subgraph group_housekeeping[Housekeeping]
        sync_gate_replica["Sync Gate Replica<br/>object / housekeeping / housekeeping"]
        checkpoint_save["Checkpoint Save<br/>object / housekeeping / housekeeping  [periodic]"]
        program_hold(["Hold / next round<br/>control / housekeeping / hold"])
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
    data_node -- "provides:pregestation_eval_loader" --> gate_0_pregestation_eval
    data_node -- "provides:gestation_loader" --> stage_1_gestation
    data_node -- "provides:gestation_eval_loader" --> gate_1_gestation_eval
    data_node -- "provides:berkeley_refresh_loader" --> stage_2_berkeley
    data_node -- "provides:gate_val_loader+payload_val_loader" --> gate_berkeley
    data_node -- "provides:payload_bank" --> stage_g_generator
    data_node -- "provides:payload_images" --> build_flashcard_rows
    stage_0_pregestation -- "after_stage0" --> gate_0_pregestation_eval
    gate_0_pregestation_eval -- "after_gate0" --> stage_1_gestation
    stage_1_gestation -- "after_stage1" --> gate_1_gestation_eval
    gate_1_gestation_eval -- "after_gate1" --> stage_2_berkeley
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
    wave_pool -- "1" --> init_vocab
    init_vocab -- "2" --> build_wave_classifier
    build_wave_classifier -- "3" --> build_classifier
    build_classifier -- "4" --> decision_004_stage_w_wave_classifier
    decision_004_stage_w_wave_classifier -- "4.1" --> stage_w_wave_classifier
    decision_004_stage_w_wave_classifier -- "4.0" --> config_search
    stage_w_wave_classifier -- "5" --> config_search
    config_search -- "6" --> vocab_churn
    vocab_churn -- "7" --> decision_007_gate_wave
    decision_007_gate_wave -- "7.1" --> gate_wave
    decision_007_gate_wave -- "7.0" --> build_transformer
    gate_wave -- "8" --> build_transformer
    build_transformer -- "9" --> build_symbol_pool
    build_symbol_pool -- "10" --> decision_010_build_gan
    decision_010_build_gan -- "10.1" --> build_gan
    decision_010_build_gan -- "10.0" --> build_label_embedding
    build_gan -- "11" --> build_label_embedding
    build_label_embedding -- "12" --> data_node
    data_node -- "13" --> stage_0_pregestation
    stage_0_pregestation -- "14" --> build_flashcard_rows
    build_flashcard_rows -- "15" --> gate_0_pregestation_eval
    gate_0_pregestation_eval -- "16" --> decision_016_stage_1_gestation
    decision_016_stage_1_gestation -- "16.1" --> stage_1_gestation
    decision_016_stage_1_gestation -- "16.0" --> gate_1_gestation_eval
    stage_1_gestation -- "17" --> decision_017_gate_1_gestation_eval
    decision_017_gate_1_gestation_eval -- "17.1" --> gate_1_gestation_eval
    decision_017_gate_1_gestation_eval -- "17.0" --> stage_2_berkeley
    gate_1_gestation_eval -- "18" --> decision_018_stage_2_berkeley
    decision_018_stage_2_berkeley -- "18.1" --> stage_2_berkeley
    decision_018_stage_2_berkeley -- "18.0" --> gate_berkeley
    stage_2_berkeley -- "19" --> decision_019_gate_berkeley
    decision_019_gate_berkeley -- "19.1" --> gate_berkeley
    decision_019_gate_berkeley -- "19.0" --> stage_r_transformer
    gate_berkeley -- "20" --> decision_020_stage_r_transformer
    decision_020_stage_r_transformer -- "20.1" --> stage_r_transformer
    decision_020_stage_r_transformer -- "20.0" --> stage_c_lora
    stage_r_transformer -- "21" --> decision_021_stage_c_lora
    decision_021_stage_c_lora -- "21.1" --> stage_c_lora
    decision_021_stage_c_lora -- "21.0" --> gate_transformer
    stage_c_lora -- "22" --> decision_022_gate_transformer
    decision_022_gate_transformer -- "22.1" --> gate_transformer
    decision_022_gate_transformer -- "22.0" --> stage_fake_feedback
    gate_transformer -- "23" --> decision_023_stage_fake_feedback
    decision_023_stage_fake_feedback -- "23.1" --> stage_fake_feedback
    decision_023_stage_fake_feedback -- "23.0" --> stage_g_generator
    stage_fake_feedback -- "24" --> decision_024_stage_g_generator
    decision_024_stage_g_generator -- "24.1" --> stage_g_generator
    decision_024_stage_g_generator -- "24.0" --> sync_gate_replica
    stage_g_generator -- "25" --> sync_gate_replica
    sync_gate_replica -- "26" --> decision_026_gate_generator
    decision_026_gate_generator -- "26.1" --> gate_generator
    decision_026_gate_generator -- "26.0" --> checkpoint_save
    gate_generator -- "27" --> checkpoint_save

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
    class gate_0_pregestation_eval,gate_1_gestation_eval,gate_berkeley,gate_transformer,gate_generator,gate_wave,decision_004_stage_w_wave_classifier,decision_007_gate_wave,decision_010_build_gan,decision_016_stage_1_gestation,decision_017_gate_1_gestation_eval,decision_018_stage_2_berkeley,decision_019_gate_berkeley,decision_020_stage_r_transformer,decision_021_stage_c_lora,decision_022_gate_transformer,decision_023_stage_fake_feedback,decision_024_stage_g_generator,decision_026_gate_generator faculty_gates;
    class sync_gate_replica,checkpoint_save,program_hold faculty_housekeeping;
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
    linkStyle 16 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 17 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 18 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 19 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 20 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 21 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 22 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 23 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 24 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 25 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 26 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 27 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 28 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 29 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 30 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 31 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 32 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 33 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 34 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 35 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 36 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 37 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 38 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 39 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 40 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 41 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 42 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 43 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 44 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 45 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 46 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 47 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 48 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 49 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 50 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 51 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 52 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 53 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 54 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 55 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 56 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 57 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 58 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 59 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 60 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 61 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 62 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 63 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 64 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 65 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 66 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 67 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 68 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 69 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 70 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 71 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 72 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 73 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 74 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 75 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 76 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 77 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 78 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 79 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 80 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 81 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 82 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 83 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 84 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 85 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
    linkStyle 86 stroke:#111111,stroke-width:2px,opacity:0.95,stroke-dasharray:6 3;
    linkStyle 87 stroke:#111111,stroke-width:3px,opacity:0.98,stroke-dasharray:0;
```

Downloads: [PNG](docs/diagrams/execution_overlay_dense.png) | [SVG](docs/diagrams/execution_overlay_dense.svg) | [MMD](docs/diagrams/execution_overlay_dense.mmd)

The colored edges retain topology and data/reaction structure. The black numbered edges show scheduler order, while diamond nodes expose the explicit guard checks that gate later stages.

**Execution layer Dense Infographic**

```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart TD
    subgraph group_bootstrap[Bootstrap]
        wave_pool["Wave Pool<br/>object / bootstrap / seed  [once]"]
    end
    subgraph group_build[Build]
        build_classifier["Build Classifier<br/>object / build / builder  [once]"]
        config_search["Config Search<br/>object / build / node  [once]"]
        build_transformer["Build Transformer<br/>object / build / builder  [once]"]
        build_gan["Build GAN<br/>object / build / builder  [once]"]
        build_wave_classifier["Build Wave Classifier<br/>object / build / builder"]
    end
    subgraph group_vocab[Vocab]
        init_vocab["Init Vocab<br/>object / vocab / seed  [once]"]
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
        stage_1_gestation["Gestation Train<br/>object / train / stage  [gated]"]
        stage_2_berkeley["Berkeley Refresh Train<br/>object / train / stage  [gated]"]
        stage_r_transformer["Transformer Train<br/>object / train / stage  [gated]"]
        stage_g_generator["Generator Train<br/>object / train / stage  [gated]"]
        stage_w_wave_classifier["Wave Classifier Train<br/>object / train / stage  [gated]"]
        stage_c_lora["LoRA Round<br/>object / train / stage  [gated]"]
        stage_fake_feedback["Fake Class Feedback<br/>object / train / stage  [gated]"]
    end
    subgraph group_gates[Gates]
        gate_0_pregestation_eval["Pregestation Eval<br/>object / gates / gate"]
        gate_1_gestation_eval["Gestation Eval<br/>object / gates / gate  [gated]"]
        gate_berkeley["Berkeley Gate<br/>object / gates / gate  [gated]"]
        gate_transformer["Transformer Gate<br/>object / gates / gate  [gated]"]
        gate_generator["Generator Gate<br/>object / gates / gate  [gated]"]
        gate_wave["Wave Gate<br/>object / gates / gate  [gated]"]
    end
    subgraph group_housekeeping[Housekeeping]
        sync_gate_replica["Sync Gate Replica<br/>object / housekeeping / housekeeping"]
        checkpoint_save["Checkpoint Save<br/>object / housekeeping / housekeeping  [periodic]"]
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
    data_node -- "provides:pregestation_eval_loader" --> gate_0_pregestation_eval
    data_node -- "provides:gestation_loader" --> stage_1_gestation
    data_node -- "provides:gestation_eval_loader" --> gate_1_gestation_eval
    data_node -- "provides:berkeley_refresh_loader" --> stage_2_berkeley
    data_node -- "provides:gate_val_loader+payload_val_loader" --> gate_berkeley
    data_node -- "provides:payload_bank" --> stage_g_generator
    data_node -- "provides:payload_images" --> build_flashcard_rows
    stage_0_pregestation -- "after_stage0" --> gate_0_pregestation_eval
    gate_0_pregestation_eval -- "after_gate0" --> stage_1_gestation
    stage_1_gestation -- "after_stage1" --> gate_1_gestation_eval
    gate_1_gestation_eval -- "after_gate1" --> stage_2_berkeley
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
    class gate_0_pregestation_eval,gate_1_gestation_eval,gate_berkeley,gate_transformer,gate_generator,gate_wave faculty_gates;
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
    linkStyle 16 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 17 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 18 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 19 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 20 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 21 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 22 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 23 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 24 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 25 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 26 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 27 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 28 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 29 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 30 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 31 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 32 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 33 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 34 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
```

Downloads: [PNG](docs/diagrams/execution_dense.png) | [SVG](docs/diagrams/execution_dense.svg) | [MMD](docs/diagrams/execution_dense.mmd)

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
    gate_0_pregestation_eval["Pregestation Eval"]
    gate_1_gestation_eval["Gestation Eval"]
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
    data_node --> gate_0_pregestation_eval
    data_node --> stage_1_gestation
    data_node --> gate_1_gestation_eval
    data_node --> stage_2_berkeley
    data_node --> gate_berkeley
    data_node --> stage_g_generator
    data_node --> build_flashcard_rows
    stage_0_pregestation --> gate_0_pregestation_eval
    gate_0_pregestation_eval --> stage_1_gestation
    stage_1_gestation --> gate_1_gestation_eval
    gate_1_gestation_eval --> stage_2_berkeley
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

Downloads: [PNG](docs/diagrams/execution_minimal.png) | [SVG](docs/diagrams/execution_minimal.svg) | [MMD](docs/diagrams/execution_minimal.mmd)

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
    gate_0_pregestation_eval["Pregestation Eval"]
    gate_1_gestation_eval["Gestation Eval"]
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
    data_node -- "provides:pregestation_eval_loader | data.provide" --> gate_0_pregestation_eval
    data_node -- "provides:gestation_loader | data.provide" --> stage_1_gestation
    data_node -- "provides:gestation_eval_loader | data.provide" --> gate_1_gestation_eval
    data_node -- "provides:berkeley_refresh_loader | data.provide" --> stage_2_berkeley
    data_node -- "provides:gate_val_loader+payload_val_loader | data.provide" --> gate_berkeley
    data_node -- "provides:payload_bank | data.provide" --> stage_g_generator
    data_node -- "provides:payload_images | data.provide" --> build_flashcard_rows
    stage_0_pregestation -- "after_stage0 | schedule.transition" --> gate_0_pregestation_eval
    gate_0_pregestation_eval -- "after_gate0 | schedule.transition" --> stage_1_gestation
    stage_1_gestation -- "after_stage1 | schedule.transition" --> gate_1_gestation_eval
    gate_1_gestation_eval -- "after_gate1 | schedule.transition" --> stage_2_berkeley
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
    linkStyle 16 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 17 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 18 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 19 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 20 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 21 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 22 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 23 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 24 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 25 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 26 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 27 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 28 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 29 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 30 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 31 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 32 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 33 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 34 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
```

Downloads: [PNG](docs/diagrams/execution_reaction.png) | [SVG](docs/diagrams/execution_reaction.svg) | [MMD](docs/diagrams/execution_reaction.mmd)

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

Downloads: [PNG](docs/diagrams/inference_dense.png) | [SVG](docs/diagrams/inference_dense.svg) | [MMD](docs/diagrams/inference_dense.mmd)

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

Downloads: [PNG](docs/diagrams/inference_minimal.png) | [SVG](docs/diagrams/inference_minimal.svg) | [MMD](docs/diagrams/inference_minimal.mmd)

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

Downloads: [PNG](docs/diagrams/inference_reaction.png) | [SVG](docs/diagrams/inference_reaction.svg) | [MMD](docs/diagrams/inference_reaction.mmd)

</details>
<!-- END:GENERATED_INFERENCE_LAYER -->

<!-- BEGIN:GENERATED_STACK_VIEW_LAYER -->
**Operations Stack View Dense Infographic**

```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    subgraph group_other[Other]
        wave_pool["Wave Pool<br/>object / other / WavePoolNode"]
        wave_pool__should_run["Gate check<br/>object / other / gate_check"]
        wave_pool__execute["Discover WAV files and decode to float32 streams<br/>object / other / execute"]
        init_vocab["Init Vocab<br/>object / other / InitVocabNode"]
        init_vocab__execute["Load and assemble initial semantic vocabulary<br/>object / other / execute"]
        build_classifier["Build Classifier<br/>object / other / BuildClassifierNode"]
        build_classifier__should_run["Gate check<br/>object / other / gate_check"]
        build_classifier__execute["Instantiate TinyConvClassifier + optimizer<br/>object / other / execute"]
        config_search["Config Search<br/>object / other / ConfigSearchNode"]
        config_search__should_run["Gate check<br/>object / other / gate_check"]
        config_search__execute["Score random RenderConfig candidates via classifier feature metric<br/>object / other / execute"]
        build_transformer["Build Transformer<br/>object / other / BuildTransformerNode"]
        build_transformer__should_run["Gate check<br/>object / other / gate_check"]
        build_transformer__execute["Instantiate WavePatchTransformer + optimizer<br/>object / other / execute"]
        build_gan["Build Gan<br/>object / other / BuildGANNode"]
        build_gan__should_run["Gate check<br/>object / other / gate_check"]
        build_gan__execute["Instantiate ConditionalBitPlaneGenerator + Discriminator<br/>object / other / execute"]
        build_wave_classifier["Build Wave Classifier<br/>object / other / BuildWaveClassifierNode"]
        build_wave_classifier__should_run["Gate check<br/>object / other / gate_check"]
        build_wave_classifier__execute["Instantiate wave-feedback TinyConvClassifier<br/>object / other / execute"]
        vocab_churn["Vocab Churn<br/>object / other / VocabChurnNode"]
        vocab_churn__should_run["Gate check<br/>object / other / gate_check"]
        vocab_churn__execute["Rotate extra vocab terms (churn) on schedule<br/>object / other / execute"]
        build_symbol_pool["Build Symbol Pool<br/>object / other / BuildSymbolPoolNode"]
        build_symbol_pool__execute["Build per-term symbol image pool<br/>object / other / execute"]
        build_label_embedding["Build Label Embedding<br/>object / other / BuildLabelEmbeddingNode"]
        build_label_embedding__should_run["Gate check<br/>object / other / gate_check"]
        build_label_embedding__execute["Build sentence-transformer label embedding bank<br/>object / other / execute"]
        build_flashcard_rows["Build Flashcard Rows<br/>object / other / BuildFlashcardRowsNode"]
        build_flashcard_rows__should_run["Gate check<br/>object / other / gate_check"]
        build_flashcard_rows__execute["Build per-term GAN conditioning flashcard rows<br/>object / other / execute"]
        data_node["Data Node<br/>object / other / DataNode"]
        data_node__should_run["Gate check<br/>object / other / gate_check"]
        data_node__execute["Build/refresh all training dataloaders<br/>object / other / execute"]
        stage_0_pregestation["Stage 0 Pregestation<br/>object / other / PregestationTrainNode"]
        stage_0_pregestation__should_run["Gate check<br/>object / other / gate_check"]
        stage_0_pregestation__execute["Stage 0: pre-gestation classifier training (geometric logic)<br/>object / other / execute"]
        stage_1_gestation["Stage 1 Gestation<br/>object / other / GestationTrainNode"]
        stage_1_gestation__execute["Stage 1: gestation classifier training (bootstrap primitives)<br/>object / other / execute"]
        stage_2_berkeley["Stage 2 Berkeley<br/>object / other / BerkeleyRefreshTrainNode"]
        stage_2_berkeley__execute["Stage 2: Berkeley SBD refresh (full multi-label classification)<br/>object / other / execute"]
        stage_r_transformer["Stage R Transformer<br/>object / other / TransformerTrainNode"]
        stage_r_transformer__should_run["Gate check<br/>object / other / gate_check"]
        stage_r_transformer__execute["Stage R: WavePatchTransformer feature-score training<br/>object / other / execute"]
        stage_g_generator["Stage G Generator<br/>object / other / GeneratorTrainNode"]
        stage_g_generator__should_run["Gate check<br/>object / other / gate_check"]
        stage_g_generator__execute["Stage G: Conditional GAN training (generator + discriminator)<br/>object / other / execute"]
        stage_w_wave_classifier["Stage W Wave Classifier<br/>object / other / WaveClassifierTrainNode"]
        stage_w_wave_classifier__should_run["Gate check<br/>object / other / gate_check"]
        stage_w_wave_classifier__execute["Stage W: Wave classifier training on transformer outputs<br/>object / other / execute"]
        stage_c_lora["Stage C Lora<br/>object / other / LoRARoundNode"]
        stage_c_lora__execute["Stage C: LoRA slot switching and per-term Berkeley training<br/>object / other / execute"]
        stage_fake_feedback["Stage Fake Feedback<br/>object / other / FakeClassFeedbackNode"]
        stage_fake_feedback__should_run["Gate check<br/>object / other / gate_check"]
        stage_fake_feedback__execute["Fake-class feedback: train classifier to detect GAN outputs<br/>object / other / execute"]
        gate_0_pregestation_eval["Gate 0 Pregestation Eval<br/>object / other / PregestationEvalNode"]
        gate_0_pregestation_eval__should_run["Gate check<br/>object / other / gate_check"]
        gate_0_pregestation_eval__execute["Gate 0 Eval: pre-gestation validation loss<br/>object / other / execute"]
        gate_1_gestation_eval["Gate 1 Gestation Eval<br/>object / other / GestationEvalNode"]
        gate_1_gestation_eval__should_run["Gate check<br/>object / other / gate_check"]
        gate_1_gestation_eval__execute["Gate 1 Eval: gestation validation loss<br/>object / other / execute"]
        gate_berkeley["Gate Berkeley<br/>object / other / BerkeleyGateNode"]
        gate_berkeley__should_run["Gate check<br/>object / other / gate_check"]
        gate_berkeley__execute["Gate 2 Eval: semantic classifier confidence + macro-F1<br/>object / other / execute"]
        gate_transformer["Gate Transformer<br/>object / other / TransformerGateNode"]
        gate_transformer__should_run["Gate check<br/>object / other / gate_check"]
        gate_transformer__execute["Gate R Eval: transformer feature score + entropy<br/>object / other / execute"]
        gate_generator["Gate Generator<br/>object / other / GeneratorGateNode"]
        gate_generator__should_run["Gate check<br/>object / other / gate_check"]
        gate_generator__execute["Gate G Eval: generator feature score<br/>object / other / execute"]
        gate_wave["Gate Wave<br/>object / other / WaveGateNode"]
        gate_wave__should_run["Gate check<br/>object / other / gate_check"]
        gate_wave__execute["Gate W Eval: wave entropy + feature score feedback<br/>object / other / execute"]
        sync_gate_replica["Sync Gate Replica<br/>object / other / SyncGateReplicaNode"]
        sync_gate_replica__should_run["Gate check<br/>object / other / gate_check"]
        sync_gate_replica__execute["Sync frozen CPU gate_classifier from main classifier<br/>object / other / execute"]
        checkpoint_save["Checkpoint Save<br/>object / other / CheckpointSaveNode"]
        checkpoint_save__should_run["Gate check<br/>object / other / gate_check"]
        checkpoint_save__execute["Save pipeline checkpoint to disk<br/>object / other / execute"]
    end

    wave_pool -- "gate_check" --> wave_pool__should_run
    wave_pool -- "execute" --> wave_pool__execute
    init_vocab -- "execute" --> init_vocab__execute
    build_classifier -- "gate_check" --> build_classifier__should_run
    build_classifier -- "execute" --> build_classifier__execute
    config_search -- "gate_check" --> config_search__should_run
    config_search -- "execute" --> config_search__execute
    build_transformer -- "gate_check" --> build_transformer__should_run
    build_transformer -- "execute" --> build_transformer__execute
    build_gan -- "gate_check" --> build_gan__should_run
    build_gan -- "execute" --> build_gan__execute
    build_wave_classifier -- "gate_check" --> build_wave_classifier__should_run
    build_wave_classifier -- "execute" --> build_wave_classifier__execute
    vocab_churn -- "gate_check" --> vocab_churn__should_run
    vocab_churn -- "execute" --> vocab_churn__execute
    build_symbol_pool -- "execute" --> build_symbol_pool__execute
    build_label_embedding -- "gate_check" --> build_label_embedding__should_run
    build_label_embedding -- "execute" --> build_label_embedding__execute
    build_flashcard_rows -- "gate_check" --> build_flashcard_rows__should_run
    build_flashcard_rows -- "execute" --> build_flashcard_rows__execute
    data_node -- "gate_check" --> data_node__should_run
    data_node -- "execute" --> data_node__execute
    stage_0_pregestation -- "gate_check" --> stage_0_pregestation__should_run
    stage_0_pregestation -- "execute" --> stage_0_pregestation__execute
    stage_1_gestation -- "execute" --> stage_1_gestation__execute
    stage_2_berkeley -- "execute" --> stage_2_berkeley__execute
    stage_r_transformer -- "gate_check" --> stage_r_transformer__should_run
    stage_r_transformer -- "execute" --> stage_r_transformer__execute
    stage_g_generator -- "gate_check" --> stage_g_generator__should_run
    stage_g_generator -- "execute" --> stage_g_generator__execute
    stage_w_wave_classifier -- "gate_check" --> stage_w_wave_classifier__should_run
    stage_w_wave_classifier -- "execute" --> stage_w_wave_classifier__execute
    stage_c_lora -- "execute" --> stage_c_lora__execute
    stage_fake_feedback -- "gate_check" --> stage_fake_feedback__should_run
    stage_fake_feedback -- "execute" --> stage_fake_feedback__execute
    gate_0_pregestation_eval -- "gate_check" --> gate_0_pregestation_eval__should_run
    gate_0_pregestation_eval -- "execute" --> gate_0_pregestation_eval__execute
    gate_1_gestation_eval -- "gate_check" --> gate_1_gestation_eval__should_run
    gate_1_gestation_eval -- "execute" --> gate_1_gestation_eval__execute
    gate_berkeley -- "gate_check" --> gate_berkeley__should_run
    gate_berkeley -- "execute" --> gate_berkeley__execute
    gate_transformer -- "gate_check" --> gate_transformer__should_run
    gate_transformer -- "execute" --> gate_transformer__execute
    gate_generator -- "gate_check" --> gate_generator__should_run
    gate_generator -- "execute" --> gate_generator__execute
    gate_wave -- "gate_check" --> gate_wave__should_run
    gate_wave -- "execute" --> gate_wave__execute
    sync_gate_replica -- "gate_check" --> sync_gate_replica__should_run
    sync_gate_replica -- "execute" --> sync_gate_replica__execute
    checkpoint_save -- "gate_check" --> checkpoint_save__should_run
    checkpoint_save -- "execute" --> checkpoint_save__execute
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
    data_node -- "provides:pregestation_eval_loader" --> gate_0_pregestation_eval
    data_node -- "provides:gestation_loader" --> stage_1_gestation
    data_node -- "provides:gestation_eval_loader" --> gate_1_gestation_eval
    data_node -- "provides:berkeley_refresh_loader" --> stage_2_berkeley
    data_node -- "provides:gate_val_loader+payload_val_loader" --> gate_berkeley
    data_node -- "provides:payload_bank" --> stage_g_generator
    data_node -- "provides:payload_images" --> build_flashcard_rows
    stage_0_pregestation -- "after_stage0" --> gate_0_pregestation_eval
    gate_0_pregestation_eval -- "after_gate0" --> stage_1_gestation
    stage_1_gestation -- "after_stage1" --> gate_1_gestation_eval
    gate_1_gestation_eval -- "after_gate1" --> stage_2_berkeley
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
    class wave_pool,wave_pool__should_run,wave_pool__execute,init_vocab,init_vocab__execute,build_classifier,build_classifier__should_run,build_classifier__execute,config_search,config_search__should_run,config_search__execute,build_transformer,build_transformer__should_run,build_transformer__execute,build_gan,build_gan__should_run,build_gan__execute,build_wave_classifier,build_wave_classifier__should_run,build_wave_classifier__execute,vocab_churn,vocab_churn__should_run,vocab_churn__execute,build_symbol_pool,build_symbol_pool__execute,build_label_embedding,build_label_embedding__should_run,build_label_embedding__execute,build_flashcard_rows,build_flashcard_rows__should_run,build_flashcard_rows__execute,data_node,data_node__should_run,data_node__execute,stage_0_pregestation,stage_0_pregestation__should_run,stage_0_pregestation__execute,stage_1_gestation,stage_1_gestation__execute,stage_2_berkeley,stage_2_berkeley__execute,stage_r_transformer,stage_r_transformer__should_run,stage_r_transformer__execute,stage_g_generator,stage_g_generator__should_run,stage_g_generator__execute,stage_w_wave_classifier,stage_w_wave_classifier__should_run,stage_w_wave_classifier__execute,stage_c_lora,stage_c_lora__execute,stage_fake_feedback,stage_fake_feedback__should_run,stage_fake_feedback__execute,gate_0_pregestation_eval,gate_0_pregestation_eval__should_run,gate_0_pregestation_eval__execute,gate_1_gestation_eval,gate_1_gestation_eval__should_run,gate_1_gestation_eval__execute,gate_berkeley,gate_berkeley__should_run,gate_berkeley__execute,gate_transformer,gate_transformer__should_run,gate_transformer__execute,gate_generator,gate_generator__should_run,gate_generator__execute,gate_wave,gate_wave__should_run,gate_wave__execute,sync_gate_replica,sync_gate_replica__should_run,sync_gate_replica__execute,checkpoint_save,checkpoint_save__should_run,checkpoint_save__execute faculty_other;
    linkStyle 0 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 1 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 2 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 3 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 4 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 5 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 6 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 7 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 8 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 9 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 10 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 11 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 12 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 13 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 14 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 15 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 16 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 17 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 18 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 19 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 20 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 21 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 22 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 23 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 24 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 25 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 26 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 27 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 28 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 29 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 30 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 31 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 32 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 33 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 34 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 35 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 36 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 37 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 38 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 39 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 40 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 41 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 42 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 43 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 44 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 45 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 46 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 47 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 48 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 49 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 50 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 51 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 52 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 53 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 54 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 55 stroke:#577590,stroke-width:3px,opacity:0.85,stroke-dasharray:8 3;
    linkStyle 56 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 57 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 58 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 59 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 60 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 61 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 62 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 63 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 64 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 65 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 66 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 67 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 68 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 69 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 70 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 71 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 72 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 73 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 74 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 75 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 76 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 77 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 78 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 79 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 80 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 81 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 82 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 83 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 84 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 85 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
```

Downloads: [PNG](docs/diagrams/stack_view_dense.png) | [SVG](docs/diagrams/stack_view_dense.svg) | [MMD](docs/diagrams/stack_view_dense.mmd)

Node colors group bootstrap, build, vocab, data, train, gate, and housekeeping faculties. Edge colors separate startup, per-round, data-provision, gated progression, and end-of-round reactions.

<details>

<summary>Minimal schematic view</summary>



```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    wave_pool["Wave Pool"]
    wave_pool__should_run["Gate check"]
    wave_pool__execute["Discover WAV files and decode to float32 streams"]
    init_vocab["Init Vocab"]
    init_vocab__execute["Load and assemble initial semantic vocabulary"]
    build_classifier["Build Classifier"]
    build_classifier__should_run["Gate check"]
    build_classifier__execute["Instantiate TinyConvClassifier + optimizer"]
    config_search["Config Search"]
    config_search__should_run["Gate check"]
    config_search__execute["Score random RenderConfig candidates via classifier feature metric"]
    build_transformer["Build Transformer"]
    build_transformer__should_run["Gate check"]
    build_transformer__execute["Instantiate WavePatchTransformer + optimizer"]
    build_gan["Build Gan"]
    build_gan__should_run["Gate check"]
    build_gan__execute["Instantiate ConditionalBitPlaneGenerator + Discriminator"]
    build_wave_classifier["Build Wave Classifier"]
    build_wave_classifier__should_run["Gate check"]
    build_wave_classifier__execute["Instantiate wave-feedback TinyConvClassifier"]
    vocab_churn["Vocab Churn"]
    vocab_churn__should_run["Gate check"]
    vocab_churn__execute["Rotate extra vocab terms (churn) on schedule"]
    build_symbol_pool["Build Symbol Pool"]
    build_symbol_pool__execute["Build per-term symbol image pool"]
    build_label_embedding["Build Label Embedding"]
    build_label_embedding__should_run["Gate check"]
    build_label_embedding__execute["Build sentence-transformer label embedding bank"]
    build_flashcard_rows["Build Flashcard Rows"]
    build_flashcard_rows__should_run["Gate check"]
    build_flashcard_rows__execute["Build per-term GAN conditioning flashcard rows"]
    data_node["Data Node"]
    data_node__should_run["Gate check"]
    data_node__execute["Build/refresh all training dataloaders"]
    stage_0_pregestation["Stage 0 Pregestation"]
    stage_0_pregestation__should_run["Gate check"]
    stage_0_pregestation__execute["Stage 0: pre-gestation classifier training (geometric logic)"]
    stage_1_gestation["Stage 1 Gestation"]
    stage_1_gestation__execute["Stage 1: gestation classifier training (bootstrap primitives)"]
    stage_2_berkeley["Stage 2 Berkeley"]
    stage_2_berkeley__execute["Stage 2: Berkeley SBD refresh (full multi-label classification)"]
    stage_r_transformer["Stage R Transformer"]
    stage_r_transformer__should_run["Gate check"]
    stage_r_transformer__execute["Stage R: WavePatchTransformer feature-score training"]
    stage_g_generator["Stage G Generator"]
    stage_g_generator__should_run["Gate check"]
    stage_g_generator__execute["Stage G: Conditional GAN training (generator + discriminator)"]
    stage_w_wave_classifier["Stage W Wave Classifier"]
    stage_w_wave_classifier__should_run["Gate check"]
    stage_w_wave_classifier__execute["Stage W: Wave classifier training on transformer outputs"]
    stage_c_lora["Stage C Lora"]
    stage_c_lora__execute["Stage C: LoRA slot switching and per-term Berkeley training"]
    stage_fake_feedback["Stage Fake Feedback"]
    stage_fake_feedback__should_run["Gate check"]
    stage_fake_feedback__execute["Fake-class feedback: train classifier to detect GAN outputs"]
    gate_0_pregestation_eval["Gate 0 Pregestation Eval"]
    gate_0_pregestation_eval__should_run["Gate check"]
    gate_0_pregestation_eval__execute["Gate 0 Eval: pre-gestation validation loss"]
    gate_1_gestation_eval["Gate 1 Gestation Eval"]
    gate_1_gestation_eval__should_run["Gate check"]
    gate_1_gestation_eval__execute["Gate 1 Eval: gestation validation loss"]
    gate_berkeley["Gate Berkeley"]
    gate_berkeley__should_run["Gate check"]
    gate_berkeley__execute["Gate 2 Eval: semantic classifier confidence + macro-F1"]
    gate_transformer["Gate Transformer"]
    gate_transformer__should_run["Gate check"]
    gate_transformer__execute["Gate R Eval: transformer feature score + entropy"]
    gate_generator["Gate Generator"]
    gate_generator__should_run["Gate check"]
    gate_generator__execute["Gate G Eval: generator feature score"]
    gate_wave["Gate Wave"]
    gate_wave__should_run["Gate check"]
    gate_wave__execute["Gate W Eval: wave entropy + feature score feedback"]
    sync_gate_replica["Sync Gate Replica"]
    sync_gate_replica__should_run["Gate check"]
    sync_gate_replica__execute["Sync frozen CPU gate_classifier from main classifier"]
    checkpoint_save["Checkpoint Save"]
    checkpoint_save__should_run["Gate check"]
    checkpoint_save__execute["Save pipeline checkpoint to disk"]

    wave_pool --> wave_pool__should_run
    wave_pool --> wave_pool__execute
    init_vocab --> init_vocab__execute
    build_classifier --> build_classifier__should_run
    build_classifier --> build_classifier__execute
    config_search --> config_search__should_run
    config_search --> config_search__execute
    build_transformer --> build_transformer__should_run
    build_transformer --> build_transformer__execute
    build_gan --> build_gan__should_run
    build_gan --> build_gan__execute
    build_wave_classifier --> build_wave_classifier__should_run
    build_wave_classifier --> build_wave_classifier__execute
    vocab_churn --> vocab_churn__should_run
    vocab_churn --> vocab_churn__execute
    build_symbol_pool --> build_symbol_pool__execute
    build_label_embedding --> build_label_embedding__should_run
    build_label_embedding --> build_label_embedding__execute
    build_flashcard_rows --> build_flashcard_rows__should_run
    build_flashcard_rows --> build_flashcard_rows__execute
    data_node --> data_node__should_run
    data_node --> data_node__execute
    stage_0_pregestation --> stage_0_pregestation__should_run
    stage_0_pregestation --> stage_0_pregestation__execute
    stage_1_gestation --> stage_1_gestation__execute
    stage_2_berkeley --> stage_2_berkeley__execute
    stage_r_transformer --> stage_r_transformer__should_run
    stage_r_transformer --> stage_r_transformer__execute
    stage_g_generator --> stage_g_generator__should_run
    stage_g_generator --> stage_g_generator__execute
    stage_w_wave_classifier --> stage_w_wave_classifier__should_run
    stage_w_wave_classifier --> stage_w_wave_classifier__execute
    stage_c_lora --> stage_c_lora__execute
    stage_fake_feedback --> stage_fake_feedback__should_run
    stage_fake_feedback --> stage_fake_feedback__execute
    gate_0_pregestation_eval --> gate_0_pregestation_eval__should_run
    gate_0_pregestation_eval --> gate_0_pregestation_eval__execute
    gate_1_gestation_eval --> gate_1_gestation_eval__should_run
    gate_1_gestation_eval --> gate_1_gestation_eval__execute
    gate_berkeley --> gate_berkeley__should_run
    gate_berkeley --> gate_berkeley__execute
    gate_transformer --> gate_transformer__should_run
    gate_transformer --> gate_transformer__execute
    gate_generator --> gate_generator__should_run
    gate_generator --> gate_generator__execute
    gate_wave --> gate_wave__should_run
    gate_wave --> gate_wave__execute
    sync_gate_replica --> sync_gate_replica__should_run
    sync_gate_replica --> sync_gate_replica__execute
    checkpoint_save --> checkpoint_save__should_run
    checkpoint_save --> checkpoint_save__execute
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
    data_node --> gate_0_pregestation_eval
    data_node --> stage_1_gestation
    data_node --> gate_1_gestation_eval
    data_node --> stage_2_berkeley
    data_node --> gate_berkeley
    data_node --> stage_g_generator
    data_node --> build_flashcard_rows
    stage_0_pregestation --> gate_0_pregestation_eval
    gate_0_pregestation_eval --> stage_1_gestation
    stage_1_gestation --> gate_1_gestation_eval
    gate_1_gestation_eval --> stage_2_berkeley
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

Downloads: [PNG](docs/diagrams/stack_view_minimal.png) | [SVG](docs/diagrams/stack_view_minimal.svg) | [MMD](docs/diagrams/stack_view_minimal.mmd)

</details>

<details>

<summary>Reaction-colored view</summary>



```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    wave_pool["Wave Pool"]
    wave_pool__should_run["Gate check"]
    wave_pool__execute["Discover WAV files and decode to float32 streams"]
    init_vocab["Init Vocab"]
    init_vocab__execute["Load and assemble initial semantic vocabulary"]
    build_classifier["Build Classifier"]
    build_classifier__should_run["Gate check"]
    build_classifier__execute["Instantiate TinyConvClassifier + optimizer"]
    config_search["Config Search"]
    config_search__should_run["Gate check"]
    config_search__execute["Score random RenderConfig candidates via classifier feature metric"]
    build_transformer["Build Transformer"]
    build_transformer__should_run["Gate check"]
    build_transformer__execute["Instantiate WavePatchTransformer + optimizer"]
    build_gan["Build Gan"]
    build_gan__should_run["Gate check"]
    build_gan__execute["Instantiate ConditionalBitPlaneGenerator + Discriminator"]
    build_wave_classifier["Build Wave Classifier"]
    build_wave_classifier__should_run["Gate check"]
    build_wave_classifier__execute["Instantiate wave-feedback TinyConvClassifier"]
    vocab_churn["Vocab Churn"]
    vocab_churn__should_run["Gate check"]
    vocab_churn__execute["Rotate extra vocab terms (churn) on schedule"]
    build_symbol_pool["Build Symbol Pool"]
    build_symbol_pool__execute["Build per-term symbol image pool"]
    build_label_embedding["Build Label Embedding"]
    build_label_embedding__should_run["Gate check"]
    build_label_embedding__execute["Build sentence-transformer label embedding bank"]
    build_flashcard_rows["Build Flashcard Rows"]
    build_flashcard_rows__should_run["Gate check"]
    build_flashcard_rows__execute["Build per-term GAN conditioning flashcard rows"]
    data_node["Data Node"]
    data_node__should_run["Gate check"]
    data_node__execute["Build/refresh all training dataloaders"]
    stage_0_pregestation["Stage 0 Pregestation"]
    stage_0_pregestation__should_run["Gate check"]
    stage_0_pregestation__execute["Stage 0: pre-gestation classifier training (geometric logic)"]
    stage_1_gestation["Stage 1 Gestation"]
    stage_1_gestation__execute["Stage 1: gestation classifier training (bootstrap primitives)"]
    stage_2_berkeley["Stage 2 Berkeley"]
    stage_2_berkeley__execute["Stage 2: Berkeley SBD refresh (full multi-label classification)"]
    stage_r_transformer["Stage R Transformer"]
    stage_r_transformer__should_run["Gate check"]
    stage_r_transformer__execute["Stage R: WavePatchTransformer feature-score training"]
    stage_g_generator["Stage G Generator"]
    stage_g_generator__should_run["Gate check"]
    stage_g_generator__execute["Stage G: Conditional GAN training (generator + discriminator)"]
    stage_w_wave_classifier["Stage W Wave Classifier"]
    stage_w_wave_classifier__should_run["Gate check"]
    stage_w_wave_classifier__execute["Stage W: Wave classifier training on transformer outputs"]
    stage_c_lora["Stage C Lora"]
    stage_c_lora__execute["Stage C: LoRA slot switching and per-term Berkeley training"]
    stage_fake_feedback["Stage Fake Feedback"]
    stage_fake_feedback__should_run["Gate check"]
    stage_fake_feedback__execute["Fake-class feedback: train classifier to detect GAN outputs"]
    gate_0_pregestation_eval["Gate 0 Pregestation Eval"]
    gate_0_pregestation_eval__should_run["Gate check"]
    gate_0_pregestation_eval__execute["Gate 0 Eval: pre-gestation validation loss"]
    gate_1_gestation_eval["Gate 1 Gestation Eval"]
    gate_1_gestation_eval__should_run["Gate check"]
    gate_1_gestation_eval__execute["Gate 1 Eval: gestation validation loss"]
    gate_berkeley["Gate Berkeley"]
    gate_berkeley__should_run["Gate check"]
    gate_berkeley__execute["Gate 2 Eval: semantic classifier confidence + macro-F1"]
    gate_transformer["Gate Transformer"]
    gate_transformer__should_run["Gate check"]
    gate_transformer__execute["Gate R Eval: transformer feature score + entropy"]
    gate_generator["Gate Generator"]
    gate_generator__should_run["Gate check"]
    gate_generator__execute["Gate G Eval: generator feature score"]
    gate_wave["Gate Wave"]
    gate_wave__should_run["Gate check"]
    gate_wave__execute["Gate W Eval: wave entropy + feature score feedback"]
    sync_gate_replica["Sync Gate Replica"]
    sync_gate_replica__should_run["Gate check"]
    sync_gate_replica__execute["Sync frozen CPU gate_classifier from main classifier"]
    checkpoint_save["Checkpoint Save"]
    checkpoint_save__should_run["Gate check"]
    checkpoint_save__execute["Save pipeline checkpoint to disk"]

    wave_pool -- "gate_check | node.owns" --> wave_pool__should_run
    wave_pool -- "execute | node.owns" --> wave_pool__execute
    init_vocab -- "execute | node.owns" --> init_vocab__execute
    build_classifier -- "gate_check | node.owns" --> build_classifier__should_run
    build_classifier -- "execute | node.owns" --> build_classifier__execute
    config_search -- "gate_check | node.owns" --> config_search__should_run
    config_search -- "execute | node.owns" --> config_search__execute
    build_transformer -- "gate_check | node.owns" --> build_transformer__should_run
    build_transformer -- "execute | node.owns" --> build_transformer__execute
    build_gan -- "gate_check | node.owns" --> build_gan__should_run
    build_gan -- "execute | node.owns" --> build_gan__execute
    build_wave_classifier -- "gate_check | node.owns" --> build_wave_classifier__should_run
    build_wave_classifier -- "execute | node.owns" --> build_wave_classifier__execute
    vocab_churn -- "gate_check | node.owns" --> vocab_churn__should_run
    vocab_churn -- "execute | node.owns" --> vocab_churn__execute
    build_symbol_pool -- "execute | node.owns" --> build_symbol_pool__execute
    build_label_embedding -- "gate_check | node.owns" --> build_label_embedding__should_run
    build_label_embedding -- "execute | node.owns" --> build_label_embedding__execute
    build_flashcard_rows -- "gate_check | node.owns" --> build_flashcard_rows__should_run
    build_flashcard_rows -- "execute | node.owns" --> build_flashcard_rows__execute
    data_node -- "gate_check | node.owns" --> data_node__should_run
    data_node -- "execute | node.owns" --> data_node__execute
    stage_0_pregestation -- "gate_check | node.owns" --> stage_0_pregestation__should_run
    stage_0_pregestation -- "execute | node.owns" --> stage_0_pregestation__execute
    stage_1_gestation -- "execute | node.owns" --> stage_1_gestation__execute
    stage_2_berkeley -- "execute | node.owns" --> stage_2_berkeley__execute
    stage_r_transformer -- "gate_check | node.owns" --> stage_r_transformer__should_run
    stage_r_transformer -- "execute | node.owns" --> stage_r_transformer__execute
    stage_g_generator -- "gate_check | node.owns" --> stage_g_generator__should_run
    stage_g_generator -- "execute | node.owns" --> stage_g_generator__execute
    stage_w_wave_classifier -- "gate_check | node.owns" --> stage_w_wave_classifier__should_run
    stage_w_wave_classifier -- "execute | node.owns" --> stage_w_wave_classifier__execute
    stage_c_lora -- "execute | node.owns" --> stage_c_lora__execute
    stage_fake_feedback -- "gate_check | node.owns" --> stage_fake_feedback__should_run
    stage_fake_feedback -- "execute | node.owns" --> stage_fake_feedback__execute
    gate_0_pregestation_eval -- "gate_check | node.owns" --> gate_0_pregestation_eval__should_run
    gate_0_pregestation_eval -- "execute | node.owns" --> gate_0_pregestation_eval__execute
    gate_1_gestation_eval -- "gate_check | node.owns" --> gate_1_gestation_eval__should_run
    gate_1_gestation_eval -- "execute | node.owns" --> gate_1_gestation_eval__execute
    gate_berkeley -- "gate_check | node.owns" --> gate_berkeley__should_run
    gate_berkeley -- "execute | node.owns" --> gate_berkeley__execute
    gate_transformer -- "gate_check | node.owns" --> gate_transformer__should_run
    gate_transformer -- "execute | node.owns" --> gate_transformer__execute
    gate_generator -- "gate_check | node.owns" --> gate_generator__should_run
    gate_generator -- "execute | node.owns" --> gate_generator__execute
    gate_wave -- "gate_check | node.owns" --> gate_wave__should_run
    gate_wave -- "execute | node.owns" --> gate_wave__execute
    sync_gate_replica -- "gate_check | node.owns" --> sync_gate_replica__should_run
    sync_gate_replica -- "execute | node.owns" --> sync_gate_replica__execute
    checkpoint_save -- "gate_check | node.owns" --> checkpoint_save__should_run
    checkpoint_save -- "execute | node.owns" --> checkpoint_save__execute
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
    data_node -- "provides:pregestation_eval_loader | data.provide" --> gate_0_pregestation_eval
    data_node -- "provides:gestation_loader | data.provide" --> stage_1_gestation
    data_node -- "provides:gestation_eval_loader | data.provide" --> gate_1_gestation_eval
    data_node -- "provides:berkeley_refresh_loader | data.provide" --> stage_2_berkeley
    data_node -- "provides:gate_val_loader+payload_val_loader | data.provide" --> gate_berkeley
    data_node -- "provides:payload_bank | data.provide" --> stage_g_generator
    data_node -- "provides:payload_images | data.provide" --> build_flashcard_rows
    stage_0_pregestation -- "after_stage0 | schedule.transition" --> gate_0_pregestation_eval
    gate_0_pregestation_eval -- "after_gate0 | schedule.transition" --> stage_1_gestation
    stage_1_gestation -- "after_stage1 | schedule.transition" --> gate_1_gestation_eval
    gate_1_gestation_eval -- "after_gate1 | schedule.transition" --> stage_2_berkeley
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

    linkStyle 0 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 1 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 2 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 3 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 4 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 5 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 6 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 7 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 8 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 9 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 10 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 11 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 12 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 13 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 14 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 15 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 16 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 17 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 18 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 19 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 20 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 21 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 22 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 23 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 24 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 25 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 26 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 27 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 28 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 29 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 30 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 31 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 32 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 33 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 34 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 35 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 36 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 37 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 38 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 39 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 40 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 41 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 42 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 43 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 44 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 45 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 46 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 47 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 48 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 49 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 50 stroke:#8B5CF6,stroke-width:2px,opacity:0.85,stroke-dasharray:4 2;
    linkStyle 51 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 52 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 53 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 54 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 55 stroke:#577590,stroke-width:3px,opacity:0.85,stroke-dasharray:8 3;
    linkStyle 56 stroke:#2A9D8F,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 57 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 58 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 59 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 60 stroke:#E9C46A,stroke-width:3px,opacity:0.9,stroke-dasharray:2 2;
    linkStyle 61 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 62 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 63 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 64 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 65 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 66 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 67 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 68 stroke:#1D70A2,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 69 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 70 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 71 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 72 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 73 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 74 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 75 stroke:#F4A261,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 76 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 77 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 78 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 79 stroke:#E76F51,stroke-width:3px,opacity:0.95,stroke-dasharray:0;
    linkStyle 80 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 81 stroke:#B56576,stroke-width:3px,opacity:0.9,stroke-dasharray:0;
    linkStyle 82 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 83 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 84 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
    linkStyle 85 stroke:#2B9348,stroke-width:3px,opacity:0.9,stroke-dasharray:4 2;
```

Downloads: [PNG](docs/diagrams/stack_view_reaction.png) | [SVG](docs/diagrams/stack_view_reaction.svg) | [MMD](docs/diagrams/stack_view_reaction.mmd)

</details>
<!-- END:GENERATED_STACK_VIEW_LAYER -->

<!-- BEGIN:GENERATED_PROVENANCE_LAYER -->
**GPU Model Provenance Dense Infographic**

```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    subgraph group_other[Other]
        build_classifier["Build Classifier<br/>object / other / BuildClassifierNode"]
        config_search["Config Search<br/>object / other / ConfigSearchNode"]
        build_label_embedding["Build Label Embedding<br/>object / other / BuildLabelEmbeddingNode"]
        stage_0_pregestation["Stage 0 Pregestation<br/>object / other / PregestationTrainNode"]
        stage_1_gestation["Stage 1 Gestation<br/>object / other / GestationTrainNode"]
        stage_2_berkeley["Stage 2 Berkeley<br/>object / other / BerkeleyRefreshTrainNode"]
        stage_r_transformer["Stage R Transformer<br/>object / other / TransformerTrainNode"]
        stage_g_generator["Stage G Generator<br/>object / other / GeneratorTrainNode"]
        stage_w_wave_classifier["Stage W Wave Classifier<br/>object / other / WaveClassifierTrainNode"]
        stage_c_lora["Stage C Lora<br/>object / other / LoRARoundNode"]
        stage_fake_feedback["Stage Fake Feedback<br/>object / other / FakeClassFeedbackNode"]
        gate_0_pregestation_eval["Gate 0 Pregestation Eval<br/>object / other / PregestationEvalNode"]
        gate_1_gestation_eval["Gate 1 Gestation Eval<br/>object / other / GestationEvalNode"]
        gate_berkeley["Gate Berkeley<br/>object / other / BerkeleyGateNode"]
        gate_transformer["Gate Transformer<br/>object / other / TransformerGateNode"]
        gate_generator["Gate Generator<br/>object / other / GeneratorGateNode"]
        gate_wave["Gate Wave<br/>object / other / WaveGateNode"]
        sync_gate_replica["Sync Gate Replica<br/>object / other / SyncGateReplicaNode"]
        build_transformer["Build Transformer<br/>object / other / BuildTransformerNode"]
        build_wave_classifier["Build Wave Classifier<br/>object / other / BuildWaveClassifierNode"]
    end
    subgraph group_inference[Inference]
        model__classifier["Classifier Model<br/>resource / inference / gpu_model"]
        model__discriminator["Discriminator Model<br/>resource / inference / gpu_model"]
        model__generator["GAN Generator<br/>resource / inference / gpu_model"]
        model__transformer["Transformer Model<br/>resource / inference / gpu_model"]
        model__wave_classifier["Wave Classifier Model<br/>resource / inference / gpu_model"]
    end

    build_classifier -- "gpu-resident classifier" --> model__classifier
    config_search -- "gpu-resident classifier" --> model__classifier
    build_label_embedding -- "gpu-resident classifier" --> model__classifier
    stage_0_pregestation -- "gpu-resident classifier" --> model__classifier
    stage_1_gestation -- "gpu-resident classifier" --> model__classifier
    stage_2_berkeley -- "gpu-resident classifier" --> model__classifier
    stage_r_transformer -- "gpu-resident classifier" --> model__classifier
    stage_g_generator -- "gpu-resident classifier" --> model__classifier
    stage_w_wave_classifier -- "gpu-resident classifier" --> model__classifier
    stage_c_lora -- "gpu-resident classifier" --> model__classifier
    stage_fake_feedback -- "gpu-resident classifier" --> model__classifier
    gate_0_pregestation_eval -- "gpu-resident classifier" --> model__classifier
    gate_1_gestation_eval -- "gpu-resident classifier" --> model__classifier
    gate_berkeley -- "gpu-resident classifier" --> model__classifier
    gate_transformer -- "gpu-resident classifier" --> model__classifier
    gate_generator -- "gpu-resident classifier" --> model__classifier
    gate_wave -- "gpu-resident classifier" --> model__classifier
    sync_gate_replica -- "gpu-resident classifier" --> model__classifier
    stage_g_generator -- "gpu-resident discriminator" --> model__discriminator
    stage_fake_feedback -- "gpu-resident discriminator" --> model__discriminator
    stage_g_generator -- "gpu-resident generator" --> model__generator
    stage_fake_feedback -- "gpu-resident generator" --> model__generator
    gate_generator -- "gpu-resident generator" --> model__generator
    build_transformer -- "gpu-resident transformer" --> model__transformer
    stage_r_transformer -- "gpu-resident transformer" --> model__transformer
    stage_w_wave_classifier -- "gpu-resident transformer" --> model__transformer
    build_wave_classifier -- "gpu-resident wave_classifier" --> model__wave_classifier
    stage_w_wave_classifier -- "gpu-resident wave_classifier" --> model__wave_classifier

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
    class model__classifier,model__discriminator,model__generator,model__transformer,model__wave_classifier faculty_inference;
    class build_classifier,config_search,build_label_embedding,stage_0_pregestation,stage_1_gestation,stage_2_berkeley,stage_r_transformer,stage_g_generator,stage_w_wave_classifier,stage_c_lora,stage_fake_feedback,gate_0_pregestation_eval,gate_1_gestation_eval,gate_berkeley,gate_transformer,gate_generator,gate_wave,sync_gate_replica,build_transformer,build_wave_classifier faculty_other;
    linkStyle 0 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 1 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 2 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 3 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 4 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 5 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 6 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 7 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 8 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 9 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 10 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 11 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 12 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 13 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 14 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 15 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 16 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 17 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 18 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 19 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 20 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 21 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 22 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 23 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 24 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 25 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 26 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 27 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
```

Downloads: [PNG](docs/diagrams/provenance_dense.png) | [SVG](docs/diagrams/provenance_dense.svg) | [MMD](docs/diagrams/provenance_dense.mmd)

Node colors group bootstrap, build, vocab, data, train, gate, and housekeeping faculties. Edge colors separate startup, per-round, data-provision, gated progression, and end-of-round reactions.

<details>

<summary>Minimal schematic view</summary>



```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    model__classifier["Classifier Model"]
    model__discriminator["Discriminator Model"]
    model__generator["GAN Generator"]
    model__transformer["Transformer Model"]
    model__wave_classifier["Wave Classifier Model"]
    build_classifier["Build Classifier"]
    config_search["Config Search"]
    build_label_embedding["Build Label Embedding"]
    stage_0_pregestation["Stage 0 Pregestation"]
    stage_1_gestation["Stage 1 Gestation"]
    stage_2_berkeley["Stage 2 Berkeley"]
    stage_r_transformer["Stage R Transformer"]
    stage_g_generator["Stage G Generator"]
    stage_w_wave_classifier["Stage W Wave Classifier"]
    stage_c_lora["Stage C Lora"]
    stage_fake_feedback["Stage Fake Feedback"]
    gate_0_pregestation_eval["Gate 0 Pregestation Eval"]
    gate_1_gestation_eval["Gate 1 Gestation Eval"]
    gate_berkeley["Gate Berkeley"]
    gate_transformer["Gate Transformer"]
    gate_generator["Gate Generator"]
    gate_wave["Gate Wave"]
    sync_gate_replica["Sync Gate Replica"]
    build_transformer["Build Transformer"]
    build_wave_classifier["Build Wave Classifier"]

    build_classifier --> model__classifier
    config_search --> model__classifier
    build_label_embedding --> model__classifier
    stage_0_pregestation --> model__classifier
    stage_1_gestation --> model__classifier
    stage_2_berkeley --> model__classifier
    stage_r_transformer --> model__classifier
    stage_g_generator --> model__classifier
    stage_w_wave_classifier --> model__classifier
    stage_c_lora --> model__classifier
    stage_fake_feedback --> model__classifier
    gate_0_pregestation_eval --> model__classifier
    gate_1_gestation_eval --> model__classifier
    gate_berkeley --> model__classifier
    gate_transformer --> model__classifier
    gate_generator --> model__classifier
    gate_wave --> model__classifier
    sync_gate_replica --> model__classifier
    stage_g_generator --> model__discriminator
    stage_fake_feedback --> model__discriminator
    stage_g_generator --> model__generator
    stage_fake_feedback --> model__generator
    gate_generator --> model__generator
    build_transformer --> model__transformer
    stage_r_transformer --> model__transformer
    stage_w_wave_classifier --> model__transformer
    build_wave_classifier --> model__wave_classifier
    stage_w_wave_classifier --> model__wave_classifier
```

Downloads: [PNG](docs/diagrams/provenance_minimal.png) | [SVG](docs/diagrams/provenance_minimal.svg) | [MMD](docs/diagrams/provenance_minimal.mmd)

</details>

<details>

<summary>Reaction-colored view</summary>



```mermaid
%%{init: {'theme':'base','flowchart':{'curve':'basis','htmlLabels':true}}}%%
flowchart LR
    model__classifier["Classifier Model"]
    model__discriminator["Discriminator Model"]
    model__generator["GAN Generator"]
    model__transformer["Transformer Model"]
    model__wave_classifier["Wave Classifier Model"]
    build_classifier["Build Classifier"]
    config_search["Config Search"]
    build_label_embedding["Build Label Embedding"]
    stage_0_pregestation["Stage 0 Pregestation"]
    stage_1_gestation["Stage 1 Gestation"]
    stage_2_berkeley["Stage 2 Berkeley"]
    stage_r_transformer["Stage R Transformer"]
    stage_g_generator["Stage G Generator"]
    stage_w_wave_classifier["Stage W Wave Classifier"]
    stage_c_lora["Stage C Lora"]
    stage_fake_feedback["Stage Fake Feedback"]
    gate_0_pregestation_eval["Gate 0 Pregestation Eval"]
    gate_1_gestation_eval["Gate 1 Gestation Eval"]
    gate_berkeley["Gate Berkeley"]
    gate_transformer["Gate Transformer"]
    gate_generator["Gate Generator"]
    gate_wave["Gate Wave"]
    sync_gate_replica["Sync Gate Replica"]
    build_transformer["Build Transformer"]
    build_wave_classifier["Build Wave Classifier"]

    build_classifier -- "gpu-resident classifier | residency.acquire" --> model__classifier
    config_search -- "gpu-resident classifier | residency.acquire" --> model__classifier
    build_label_embedding -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_0_pregestation -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_1_gestation -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_2_berkeley -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_r_transformer -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_g_generator -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_w_wave_classifier -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_c_lora -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_fake_feedback -- "gpu-resident classifier | residency.acquire" --> model__classifier
    gate_0_pregestation_eval -- "gpu-resident classifier | residency.acquire" --> model__classifier
    gate_1_gestation_eval -- "gpu-resident classifier | residency.acquire" --> model__classifier
    gate_berkeley -- "gpu-resident classifier | residency.acquire" --> model__classifier
    gate_transformer -- "gpu-resident classifier | residency.acquire" --> model__classifier
    gate_generator -- "gpu-resident classifier | residency.acquire" --> model__classifier
    gate_wave -- "gpu-resident classifier | residency.acquire" --> model__classifier
    sync_gate_replica -- "gpu-resident classifier | residency.acquire" --> model__classifier
    stage_g_generator -- "gpu-resident discriminator | residency.acquire" --> model__discriminator
    stage_fake_feedback -- "gpu-resident discriminator | residency.acquire" --> model__discriminator
    stage_g_generator -- "gpu-resident generator | residency.acquire" --> model__generator
    stage_fake_feedback -- "gpu-resident generator | residency.acquire" --> model__generator
    gate_generator -- "gpu-resident generator | residency.acquire" --> model__generator
    build_transformer -- "gpu-resident transformer | residency.acquire" --> model__transformer
    stage_r_transformer -- "gpu-resident transformer | residency.acquire" --> model__transformer
    stage_w_wave_classifier -- "gpu-resident transformer | residency.acquire" --> model__transformer
    build_wave_classifier -- "gpu-resident wave_classifier | residency.acquire" --> model__wave_classifier
    stage_w_wave_classifier -- "gpu-resident wave_classifier | residency.acquire" --> model__wave_classifier

    linkStyle 0 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 1 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 2 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 3 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 4 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 5 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 6 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 7 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 8 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 9 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 10 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 11 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 12 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 13 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 14 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 15 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 16 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 17 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 18 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 19 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 20 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 21 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 22 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 23 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 24 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 25 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 26 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
    linkStyle 27 stroke:#6B7280,stroke-width:2px,opacity:0.75,stroke-dasharray:0;
```

Downloads: [PNG](docs/diagrams/provenance_reaction.png) | [SVG](docs/diagrams/provenance_reaction.svg) | [MMD](docs/diagrams/provenance_reaction.mmd)

</details>
<!-- END:GENERATED_PROVENANCE_LAYER -->

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
