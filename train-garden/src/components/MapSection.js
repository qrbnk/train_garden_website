import React from 'react';
import { MapPin } from 'lucide-react';

const MapSection = () => {
  // Use a constant for the URL 
  const mapUrl = "https://www.google.com/maps/d/u/1/embed?mid=1r0HJw53KoNuucSnhGcNFn5v5IHZ_U6w&ehbc=2E312F"; 

  return (
    <section id="map" className="py-16 px-4 bg-white">
      <div className="container mx-auto max-w-6xl">
        
        {/* Map Header */}
        <h2 className="text-3xl font-bold text-gray-800 mb-4 text-center flex items-center justify-center">
          <MapPin className="w-6 h-6 mr-2 text-green-600" /> Skill Hubs Map
        </h2>
        <p className="text-md text-gray-600 mb-6 text-center">
          Find volunteering and sustainability projects near you!
        </p>

        {/* Map Container: Constrained to prevent full-screen and ensure responsiveness */}
        <div className="map-container rounded-xl shadow-2xl overflow-hidden border-4 border-green-600" 
             // Height constraint applied here
             style={{ height: '70vh', minHeight: '400px' }}>
          <iframe
            src={mapUrl}
            title="Skills Hubs Around You"
            loading="lazy"
            referrerPolicy="no-referrer-when-downgrade"
            // Crucial: Omission of allowFullScreen prevents browser full-screen mode.
            style={{ width: '100%', height: '100%', border: 0 }} 
          ></iframe>
        </div>
        <div className="text-center mt-4 text-xs text-gray-500">
          *Map data sourced from Google My Maps.
        </div>
      </div>
    </section>
  );
};

export default MapSection;