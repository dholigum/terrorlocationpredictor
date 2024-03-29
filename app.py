import streamlit as st
import pandas as pd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import rasterio
from sklearn.ensemble import RandomForestClassifier
import joblib

# Setting page layout
st.set_page_config(
    page_title="SISDIKSI-TEROR",  # Setting page title
    page_icon="🛰️",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

pop_data = './spatial_data/idn_pd_2020_1km_UNadj.tif'
night_light_data = './spatial_data/idn_viirs_ntl.tif'
urban_acc_data = './spatial_data/acc_50k_idn.tif'
elev_data = './spatial_data/indo_gdem_compress.tif'
ndvi_data = './spatial_data/ndvi_indonesia_raw.tif'

rf = RandomForestClassifier()

st.sidebar.title("Selamat Datang di SISDIKSI-TEROR: Sistem Prediksi Lokasi Potensi Terorisme di Indonesia")
st.sidebar.caption("dikembangkan oleh: Eca Indah Anggraini S.Si")
st.sidebar.caption("sebagai syarat memperoleh gelar Magister dalam Teknologi Penginderaan")
street = st.sidebar.text_input("Nama Jalan", "Jalan Raya Pasar Babelan")
city = st.sidebar.text_input("Kota/Kabupaten", "Bekasi")
province = st.sidebar.text_input("Provinsi", "Jawa Barat")
country = st.sidebar.text_input("Negara", "Indonesia")

geolocator = Nominatim(user_agent="GTA Lookup")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
island = ""

if st.sidebar.button('Prediksi Terorisme'):
    location = geolocator.geocode(street+", "+city+", "+province+", "+country)
    print(location.address)

    if "Jawa" in location.address:
        rf = joblib.load("./weights/rf_Jawa.joblib")
        island = "Jawa"
    elif "Sumatera" in location.address:
        rf = joblib.load("./weights/rf_Sumatera.joblib")
        island = "Sumatera"
    elif "Kalimantan" in location.address:
        rf = joblib.load("./weights/rf_Kalimantan.joblib")
        island = "Kalimantan"
    elif "Sulawesi" in location.address:
        rf = joblib.load("./weights/rf_Sulawesi.joblib")
        island = "Sulawesi"
    elif "Papua" in location.address:
        rf = joblib.load("./weights/rf_Papua.joblib")
        island = "Papua"
    else:
        st.subheader("Mohon maaf model untuk lokasi tersebut tidak tersedia ...")
        exit()

    lat = location.latitude
    lon = location.longitude

    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})

    with rasterio.open(pop_data) as src:
        # Get the population density values at the given coordinate
        row_idx, col_idx = src.index(lon, lat)
        pop_dense = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))[0][0]
    
    with rasterio.open(night_light_data) as src:
        # Get the population density values at the given coordinate
        row_idx, col_idx = src.index(lon, lat)
        night_light = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))[0][0]

    with rasterio.open(urban_acc_data) as src:
        # Get the population density values at the given coordinate
        row_idx, col_idx = src.index(lon, lat)
        urban_acc = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))[0][0]
   
    with rasterio.open(elev_data) as src:
        # Get the population density values at the given coordinate
        row_idx, col_idx = src.index(lon, lat)
        elevation = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))[0][0]
     
    with rasterio.open(ndvi_data) as src:
        # Get the population density values at the given coordinate
        row_idx, col_idx = src.index(lon, lat)
        ndvi = src.read(1, window=((row_idx, row_idx+1), (col_idx, col_idx+1)))[0][0]

    pred_res = rf.predict([[pop_dense, night_light, urban_acc, elevation, ndvi]])
    if pred_res[0] == 0:
        st.subheader("Hasil Prediksi: **:green[Lokasi Potensi Non-Terror]**")
    else:
        st.subheader("Hasil Prediksi: **:red[Lokasi Potensi Terrorisme]**")
    
    st.map(map_data, zoom=6)
    st.text("Titik koordinat: ({0}, {1})".format(lat, lon))
    st.text("Karakteristik Geospasial:")
    st.text("Kepadatan Penduduk: {:.2f} penduduk/km2".format(pop_dense))
    st.text("Indeks Cahaya Malam Hari: {:.2f} lumen".format(night_light))
    st.text("Aksesibilitas Perkotaan: {:.2f}".format(urban_acc))
    st.text("Ketinggian Tanah: {:.2f} m".format(elevation))
    st.text("NDVI: {:.2f}".format(ndvi))

else:
    map_data = pd.DataFrame({'lat': [-6.2], 'lon': [106.81]})
    st.map(map_data, zoom=6)

st.sidebar.text("Model RF yg digunakan adalah")
st.sidebar.text("*Model Pulau {0}*".format(island))
