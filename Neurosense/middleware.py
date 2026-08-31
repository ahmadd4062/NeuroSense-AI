class LanguageMiddleware:
    """
    Middleware to detect and set the user's preferred language.
    Checks session, then cookie, then defaults to 'en'.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Try to get language from session
        lang = request.session.get('language')
        
        # If not in session, try cookie
        if not lang:
            lang = request.COOKIES.get('language')
        
        # Default to English
        if not lang or lang not in ['en', 'ur']:
            lang = 'en'
        
        # Attach language to request for use in views and templates
        request.LANG = lang
        
        response = self.get_response(request)
        return response