#!/bin/bash
# 🥓 BACON ECOSYSTEM - Complete Setup Script

echo "🥓🔥 BACON ECOSYSTEM SETUP 🔥🥓"
echo "=================================="
echo ""

# 1. Create structure
echo "📁 Creating directory structure..."
mkdir -p bacon-ecosystem/{bacon-station-v3,baconalgo-web,shared}

# 2. Setup Bacon Station V3
echo "🥓 Setting up Bacon Station V3..."
cd bacon-ecosystem/bacon-station-v3

cat > requirements.txt << 'EOF'
streamlit==1.29.0
pandas==2.1.4
ib_insync==0.9.86
supabase==2.0.3
python-dotenv==1.0.0
plotly==5.18.0
EOF

python -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

# Create .env template
cat > .env << 'EOF'
SUPABASE_URL=https://aqicrgckmuqavbxxlwl.supabase.co
SUPABASE_ANON_KEY=PASTE_YOUR_KEY_HERE
EOF

echo "✅ Bacon Station V3 ready!"

# 3. Setup BaconAlgo Web
echo "🌐 Setting up BaconAlgo.com..."
cd ../baconalgo-web

cat > requirements.txt << 'EOF'
streamlit==1.29.0
supabase==2.0.3
python-dotenv==1.0.0
pandas==2.1.4
plotly==5.18.0
EOF

python -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

# Create .env template
cat > .env << 'EOF'
SUPABASE_URL=https://aqicrgckmuqavbxxlwl.supabase.co
SUPABASE_ANON_KEY=PASTE_YOUR_KEY_HERE
EOF

echo "✅ BaconAlgo.com ready!"

cd ../..

echo ""
echo "🎉 SETUP COMPLETE!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env files with your Supabase key"
echo "2. Run Bacon Station V3:"
echo "   cd bacon-ecosystem/bacon-station-v3"
echo "   streamlit run bacon_station_v3_ultimate.py"
echo ""
echo "3. Run BaconAlgo.com:"
echo "   cd bacon-ecosystem/baconalgo-web"
echo "   streamlit run app.py"
