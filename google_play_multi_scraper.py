#!/usr/bin/env python3
"""
Google Play Store Multi-App Scraper
Scrapes app information and user reviews for multiple apps and saves to CSV and XLSX formats.
"""

import pandas as pd
import numpy as np
from google_play_scraper import app, reviews, Sort
import time
import json
from datetime import datetime
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GooglePlayMultiScraper:
    """Scraper for multiple Google Play Store apps."""
    
    def __init__(self, app_ids: List[str], country: str = 'id', lang: str = 'id'):
        self.app_ids = app_ids
        self.country = country
        self.lang = lang
        self.apps_df = None
        self.reviews_df = None
        
    def get_app_info(self, app_id: str) -> Dict:
        """Get detailed app information from Google Play Store."""
        try:
            app_info = app(app_id, lang=self.lang, country=self.country)
            
            return {
                'app_id': app_id,
                'title': app_info.get('title', ''),
                'developer': app_info.get('developer', ''),
                'developer_id': app_info.get('developerId', ''),
                'category': app_info.get('genre', ''),
                'rating': app_info.get('score', 0),
                'rating_count': app_info.get('ratings', 0),
                'installs': app_info.get('installs', ''),
                'price': app_info.get('price', 0),
                'free': app_info.get('free', True),
                'size': app_info.get('size', ''),
                'min_android': app_info.get('minInstalls', ''),
                'content_rating': app_info.get('contentRating', ''),
                'description': app_info.get('description', ''),
                'summary': app_info.get('summary', ''),
                'updated': app_info.get('updated', ''),
                'version': app_info.get('version', ''),
                'recent_changes': app_info.get('recentChanges', ''),
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting app info for {app_id}: {e}")
            return {
                'app_id': app_id,
                'error': str(e),
                'scraped_at': datetime.now().isoformat()
            }
    
    def scrape_app_reviews(self, app_id: str, count: int = 1000) -> List[Dict]:
        """Scrape reviews for a specific app."""
        print(f"\nScraping {count} reviews for {app_id}...")
        
        try:
            all_reviews = []
            continuation_token = None
            batch_size = 200
            
            with tqdm(total=count, desc=f"Reviews for {app_id}") as pbar:
                while len(all_reviews) < count:
                    try:
                        result, continuation_token = reviews(
                            app_id,
                            lang=self.lang,
                            country=self.country,
                            sort=Sort.NEWEST,
                            count=min(batch_size, count - len(all_reviews)),
                            continuation_token=continuation_token
                        )
                        
                        if not result:
                            print(f"No more reviews available for {app_id}")
                            break
                        
                        # Add app_id to each review
                        for review in result:
                            review['app_id'] = app_id
                            review['scraped_at'] = datetime.now().isoformat()
                        
                        all_reviews.extend(result)
                        pbar.update(len(result))
                        
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        print(f"Error scraping batch for {app_id}: {e}")
                        break
            
            print(f"Successfully scraped {len(all_reviews)} reviews for {app_id}")
            return all_reviews
            
        except Exception as e:
            print(f"Error scraping reviews for {app_id}: {e}")
            return []
    
    def scrape_all_apps(self, reviews_per_app: int = 1000):
        """Scrape information and reviews for all apps."""
        print("="*60)
        print("GOOGLE PLAY STORE MULTI-APP SCRAPER")
        print("="*60)
        print(f"Apps to scrape: {len(self.app_ids)}")
        print(f"Reviews per app: {reviews_per_app}")
        print(f"App IDs: {self.app_ids}")
        
        # Scrape app information
        print("\n1. Scraping app information...")
        apps_info = []
        
        for app_id in tqdm(self.app_ids, desc="Getting app info"):
            info = self.get_app_info(app_id)
            apps_info.append(info)
            print(f"✓ {info.get('title', app_id)} - Rating: {info.get('rating', 'N/A')}")
            time.sleep(1)
        
        self.apps_df = pd.DataFrame(apps_info)
        print(f"App information scraped for {len(self.apps_df)} apps")
        
        # Scrape reviews
        print("\n2. Scraping reviews...")
        all_reviews = []
        
        for app_id in self.app_ids:
            app_reviews = self.scrape_app_reviews(app_id, reviews_per_app)
            all_reviews.extend(app_reviews)
            print(f"Total reviews collected so far: {len(all_reviews)}")
            time.sleep(3)  # Longer pause between apps
        
        if all_reviews:
            self.reviews_df = pd.DataFrame(all_reviews)
            
            # Add additional features
            self.reviews_df['review_length'] = self.reviews_df['content'].str.len()
            self.reviews_df['word_count'] = self.reviews_df['content'].str.split().str.len()
            self.reviews_df['at'] = pd.to_datetime(self.reviews_df['at'])
            
            print(f"Reviews DataFrame created with {len(self.reviews_df)} rows")
        else:
            print("No reviews were scraped!")
            self.reviews_df = pd.DataFrame()
    
    def export_data(self, timestamp: str = None):
        """Export scraped data to CSV and XLSX formats."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n3. Exporting data...")
        
        # Export app information
        if not self.apps_df.empty:
            apps_csv = f"google_play_apps_info_{timestamp}.csv"
            apps_xlsx = f"google_play_apps_info_{timestamp}.xlsx"
            
            self.apps_df.to_csv(apps_csv, index=False, encoding='utf-8')
            print(f"✓ App information saved to: {apps_csv}")
            
            try:
                self.apps_df.to_excel(apps_xlsx, index=False, engine='openpyxl')
                print(f"✓ App information saved to: {apps_xlsx}")
            except ImportError:
                print("⚠️  openpyxl not installed. Install with: pip install openpyxl")
        
        # Export reviews
        if not self.reviews_df.empty:
            reviews_csv = f"google_play_reviews_{timestamp}.csv"
            reviews_xlsx = f"google_play_reviews_{timestamp}.xlsx"
            
            self.reviews_df.to_csv(reviews_csv, index=False, encoding='utf-8')
            print(f"✓ Reviews saved to: {reviews_csv}")
            
            try:
                self.reviews_df.to_excel(reviews_xlsx, index=False, engine='openpyxl')
                print(f"✓ Reviews saved to: {reviews_xlsx}")
            except ImportError:
                print("⚠️  openpyxl not installed. Install with: pip install openpyxl")
        
        # Create combined Excel file
        self.create_combined_excel(timestamp)
        
        return timestamp
    
    def create_combined_excel(self, timestamp: str):
        """Create a combined Excel file with multiple sheets."""
        try:
            combined_filename = f"google_play_complete_data_{timestamp}.xlsx"
            
            with pd.ExcelWriter(combined_filename, engine='openpyxl') as writer:
                if not self.apps_df.empty:
                    self.apps_df.to_excel(writer, sheet_name='App_Information', index=False)
                
                if not self.reviews_df.empty:
                    self.reviews_df.to_excel(writer, sheet_name='Reviews', index=False)
                    
                    # Create summary sheet
                    summary_data = self.create_summary_data()
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False, header=False)
            
            print(f"🎉 Combined Excel file created: {combined_filename}")
            
        except ImportError:
            print("⚠️  openpyxl not installed. Combined Excel file not created.")
        except Exception as e:
            print(f"Error creating combined Excel file: {e}")
    
    def create_summary_data(self) -> List[List]:
        """Create summary data for the summary sheet."""
        summary_data = []
        
        summary_data.append(['Metric', 'Value'])
        summary_data.append(['Total Apps', len(self.apps_df) if not self.apps_df.empty else 0])
        summary_data.append(['Total Reviews', len(self.reviews_df) if not self.reviews_df.empty else 0])
        
        if not self.reviews_df.empty:
            summary_data.append(['Date Range Start', self.reviews_df['at'].min().strftime('%Y-%m-%d')])
            summary_data.append(['Date Range End', self.reviews_df['at'].max().strftime('%Y-%m-%d')])
            summary_data.append(['Average Rating', round(self.reviews_df['score'].mean(), 2)])
            summary_data.append(['', ''])
            
            # Rating distribution
            summary_data.append(['Rating Distribution', ''])
            rating_dist = self.reviews_df['score'].value_counts().sort_index()
            for rating, count in rating_dist.items():
                percentage = (count / len(self.reviews_df)) * 100
                summary_data.append([f'{rating} Stars', f'{count} ({percentage:.1f}%)'])
        
        return summary_data
    
    def print_summary(self):
        """Print a summary of the scraped data."""
        print("\n" + "="*60)
        print("SCRAPING SUMMARY")
        print("="*60)
        
        if not self.apps_df.empty:
            print(f"\n✅ App Information:")
            print(f"   • {len(self.apps_df)} apps scraped")
            for _, app in self.apps_df.iterrows():
                if 'title' in app and 'rating' in app:
                    print(f"   • {app['title']}: {app['rating']:.1f}★")
        
        if not self.reviews_df.empty:
            print(f"\n✅ Reviews:")
            print(f"   • {len(self.reviews_df)} total reviews")
            print(f"   • Average rating: {self.reviews_df['score'].mean():.1f}")
            print(f"   • Date range: {(self.reviews_df['at'].max() - self.reviews_df['at'].min()).days} days")
            
            print("\n📊 Reviews by app:")
            app_counts = self.reviews_df['app_id'].value_counts()
            for app_id, count in app_counts.items():
                app_name = self.apps_df[self.apps_df['app_id'] == app_id]['title'].iloc[0] if not self.apps_df.empty else app_id
                print(f"   • {app_name}: {count} reviews")

def main():
    """Main function to run the multi-app scraper."""
    # Define app IDs to scrape
    app_ids = [
        # 'com.whatsapp',
        # 'com.facebook.katana',
        # 'com.instagram.android',
        # 'com.snapchat.android',
        # 'com.spotify.music'
        'com.jago.digitalBanking', #bank jago
        'id.co.bankbkemobile.digitalbank', #seabank
        'com.btpn.dc', #btpn jenius
        'com.bcadigital.blu', #bank bca
        'com.bnc.finance' #neobank
    ]
    # app_packages = [
    # # 'com.alloapp.yump', #allo bank
    # # 'com.senyumkubank.rekeningonline', #amarbank
    # # 'id.aladinbank.mobile',
    # # 'id.co.bankraya.apps', #rayabank
    # #   'com.supercell.brawlstars',
    # #   'jp.pokemon.pokemonunite',
    # ]
    
    # Configuration
    COUNTRY = 'id'  # Indonesia
    LANG = 'id'     # Indonesian
    REVIEWS_PER_APP = 2000
    
    # Initialize scraper
    scraper = GooglePlayMultiScraper(app_ids, COUNTRY, LANG)
    
    try:
        # Scrape all data
        scraper.scrape_all_apps(REVIEWS_PER_APP)
        
        # Export data
        timestamp = scraper.export_data()
        
        # Print summary
        scraper.print_summary()
        
        print(f"\n🎉 Scraping completed successfully!")
        print(f"📁 Files saved with timestamp: {timestamp}")
        print("\n💡 Next steps:")
        print("   • Open the Excel files to explore the data")
        print("   • Use the CSV files for further analysis")
        print("   • Consider running sentiment analysis on reviews")
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
