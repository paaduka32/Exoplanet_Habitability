# Exoplanet Habitability 

## Overview

This project explores the factors influencing the habitability of exoplanets by testing various hypotheses related to planetary properties, orbital mechanics, and stellar influences. The analysis uses observational data and statistical methods to evaluate key relationships affecting the potential for life on exoplanets.

## Hypotheses Tested

1. **Mass-Radius Relationship**: The relationship between mass and radius of exoplanets can influence their potential habitability.  
2. **Composition & Habitability**: The composition of habitable exoplanets plays a crucial role in determining their potential to support life.  
3. **Water Presence & Temperature**: The presence of water on exoplanets depends on equilibrium temperature ranges suitable for habitability.  
4. **Orbital Period Suitability**: Exoplanets with orbital periods between 50-500 days are most suitable for habitability due to optimal energy reception.  
5. **Orbital Stability**: The semi-major axis and Hill sphere influence habitability by determining orbital stability and gravitational dominance.  
6. **Atmospheric Retention**: Atmospheric retention, influenced by escape velocity, is crucial for maintaining surface conditions suitable for habitability.  
7. **Climate Stability**: Orbital distance and equilibrium temperature jointly affect habitability by regulating climate stability and surface conditions.  
8. **Tidal Effects**: Orbital period and tidal locking impact habitability by influencing temperature and atmospheric dynamics on exoplanets.  
9. **Host Star Metallicity**: The metallicity of host stars differs for terrestrial and non-terrestrial exoplanets, influencing their formation and composition.  
10. **Eccentricity & Temperature**: Planetary eccentricity affects surface temperature by causing variations in stellar energy received throughout the orbit.  
11. **Stellar Flux & Habitability**: Stellar flux directly influences the surface temperature of exoplanets, affecting their potential habitability.  
12. **Eccentricity & Habitability**: Higher orbital eccentricity negatively impacts planetary habitability by causing extreme temperature variations.  

## Project Structure
📂 Exoplanet-Habitability
│── 📂 static
│ ├── background.png
│ ├── script.js
│ ├── styles.css
│── 📄 README.md # Project documentation
│── exoplanet_dataset.csv
|── habitability.ipynb # Reference notebook for analysis
│── data_processing.py # Data cleaning and preprocessing
│── app.py

## Usage
Run the following script to clean and preprocess the data:
   ```bash
python scripts/data_processing.py
```
To start the Dash app for visualization:
```bash
python app/app.py
```

